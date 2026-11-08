"""Mã dùng chung cho Giai đoạn 2 (đề tài: hệ hỗ trợ quyết định nguy cơ tiền tiểu đường/tiểu đường, BRFSS 2015).

Nội dung: đọc dữ liệu và danh sách tập cố định, xây dựng mô hình, tinh chỉnh siêu tham số,
chỉ số đánh giá, chọn ngưỡng, hiệu chuẩn xác suất, kiểm định thống kê.

Nguyên tắc chống rò rỉ:
- Tập test chỉ được dùng ở bước đánh giá cuối (evaluate_final), sau khi cấu hình đã được chọn bằng validation.
- Tinh chỉnh siêu tham số chỉ dùng tập train (cross-validation theo nhóm hồ sơ).
- Bộ hiệu chuẩn và ngưỡng chỉ được học trên validation (hiệu chuẩn dùng cross-fitting khi đánh giá trên validation).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, log_loss, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------
# Hằng số dữ liệu
# ----------------------------------------------------------------------------------------------
CSV_NAME = "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"
FEATURES_21 = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
NUMERIC_COLS = ["BMI", "MentHlth", "PhysHlth"]  # định lượng; 18 biến còn lại là nhị phân/thứ bậc (mã rời rạc)
CAT_IDX = [i for i, c in enumerate(FEATURES_21) if c not in NUMERIC_COLS]  # chỉ số cột rời rạc cho SMOTENC
SMOTE_RATIO = 0.5  # SMOTENC nâng lớp 1 lên 50% số lớp 0 (không cân bằng hoàn toàn)
STRATEGIES = ["none", "class_weight", "smotenc"]
STRATEGY_LABEL = {"none": "Không cân bằng", "class_weight": "class_weight / scale_pos_weight", "smotenc": "SMOTENC (chỉ trên train)"}


def find_project_dir() -> Path:
    env = os.environ.get("DSS_PROJECT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for cand in [cwd, *list(cwd.parents)[:2]]:
        if (cand / "datas").is_dir() or (cand / "DSS_FinalProject.ipynb").is_file():
            return cand
    return cwd


FOLDER = find_project_dir()
DATA = FOLDER / "datas"
SPLIT_DIR = DATA / "splits"
RESULTS = FOLDER / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PRED_DIR = RESULTS / "predictions"
MODEL_DIR = FOLDER / "models"
for _p in [DATA, SPLIT_DIR, TABLES, FIGURES, PRED_DIR, MODEL_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


def save_table(table: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = TABLES / f"{name}.csv"
    table.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"-> Đã lưu bảng: {path}")
    return path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Dữ liệu và danh sách tập
# ----------------------------------------------------------------------------------------------
_CACHE: dict = {}


def _find_csv() -> Path:
    for p in [DATA / CSV_NAME, FOLDER / CSV_NAME, FOLDER / "data" / CSV_NAME]:
        if p.exists():
            return p
    raise FileNotFoundError(f"Không tìm thấy {CSV_NAME}. Hãy đặt file vào thư mục: {DATA}")


def get_data():
    """Trả về (df gồm 21 đặc trưng + nhãn, mảng profile_id). profile_id giống Giai đoạn 1."""
    if "df" not in _CACHE:
        path = _find_csv()
        raw = pd.read_csv(path)
        assert list(raw.columns) == [TARGET] + FEATURES_21, "File dữ liệu không đúng cấu trúc BRFSS 2015"
        df = raw[FEATURES_21 + [TARGET]].astype(int).reset_index(drop=True)
        pid = df.groupby(FEATURES_21, sort=False).ngroup().to_numpy()
        _CACHE.update(df=df, pid=pid, path=path)
        manifest_path = SPLIT_DIR / "manifest.json"
        if manifest_path.exists():
            man = json.load(open(manifest_path, encoding="utf-8"))
            assert sha256_of(path) == man["data_sha256"], "File dữ liệu khác với file lúc tạo danh sách tập (Giai đoạn 1)"
    return _CACHE["df"], _CACHE["pid"]


def make_split_labels(scenario: str, seed: int) -> np.ndarray:
    """scenario='profile': chia theo hồ sơ (phân tầng theo loại nhãn x cỡ hồ sơ); 'record': ngẫu nhiên theo bản ghi."""
    df, pid = get_data()
    y = df[TARGET].to_numpy()
    if scenario == "record":
        idx = np.arange(len(df))
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
        out = np.empty(len(idx), dtype=object)
        out[tr], out[va], out[te] = "train", "validation", "test"
        return out
    if scenario != "profile":
        raise ValueError(scenario)
    prof = pd.DataFrame({"pid": pid, "y": y}).groupby("pid")["y"].agg(n="size", pos="sum")
    n, pos = prof["n"].to_numpy(), prof["pos"].to_numpy()
    label_type = np.where(pos == 0, 0, np.where(pos == n, 1, 2))
    strata = label_type * 10 + np.minimum(n, 4)
    ids = prof.index.to_numpy()
    tr, tmp, _, st_tmp = train_test_split(ids, strata, test_size=0.30, stratify=strata, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=st_tmp, random_state=seed)
    return np.where(np.isin(pid, tr), "train", np.where(np.isin(pid, va), "validation", "test"))


def get_split(scenario: str, seed: int, max_train_rows: int | None = None):
    """Đọc danh sách tập đã lưu (hoặc tạo và lưu nếu seed mới). Trả về dict {'train'|'validation'|'test': (X, y, groups)}.
    max_train_rows: chỉ dùng cho chạy thử nhanh; lấy mẫu ngẫu nhiên theo hồ sơ từ train."""
    df, pid = get_data()
    path = SPLIT_DIR / f"split_by_{scenario}_seed{seed}.csv"
    if path.exists():
        sp = pd.read_csv(path)
        assert len(sp) == len(df) and (sp["row_id"].to_numpy() == np.arange(len(df))).all()
        labels = sp["split"].to_numpy()
    else:
        labels = make_split_labels(scenario, seed)
        pd.DataFrame({"row_id": np.arange(len(df)), "profile_id": pid, "split": labels}).to_csv(path, index=False)
        print(f"-> Đã tạo và lưu danh sách tập: {path}")
    out = {}
    for k in ["train", "validation", "test"]:
        m = labels == k
        out[k] = (df.loc[m, FEATURES_21].reset_index(drop=True), df.loc[m, TARGET].reset_index(drop=True), pid[m])
    if max_train_rows is not None and len(out["train"][1]) > max_train_rows:
        rng = np.random.RandomState(seed)
        keep = np.sort(rng.choice(len(out["train"][1]), max_train_rows, replace=False))
        X, y, g = out["train"]
        out["train"] = (X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True), g[keep])
    return out


# ----------------------------------------------------------------------------------------------
# Mô hình
# ----------------------------------------------------------------------------------------------
def available_models(include_ebm: bool = False):
    names, missing = ["Logistic Regression", "Random Forest"], []
    for label, mod in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
        try:
            __import__(mod)
            names.append(label)
        except ImportError:
            missing.append(label)
    if include_ebm:
        try:
            __import__("interpret")
            names.append("EBM")
        except ImportError:
            missing.append("EBM")
    return names, missing


def supported_strategies(name: str):
    return ["none"] if name == "EBM" else list(STRATEGIES)


def smotenc_resample(X: pd.DataFrame, y, ratio: float = SMOTE_RATIO, k: int = 5, seed: int = 0):
    """SMOTENC tự cài đặt (không dùng thư viện imbalanced-learn), theo Chawla et al. (2002):
    - lớp thiểu số (y = 1) được nâng lên `ratio` x số lớp 0 bằng các mẫu tổng hợp;
    - khoảng cách láng giềng: đặc trưng số (BMI, MentHlth, PhysHlth) chuẩn hóa theo độ lệch chuẩn của lớp thiểu số; mỗi đặc trưng rời rạc
      được mã hóa one-hot với trọng số = trung vị độ lệch chuẩn của các đặc trưng số / 2 (nên hai giá trị rời rạc khác nhau bị phạt tương ứng);
    - mẫu mới: đặc trưng số nội suy giữa mẫu gốc và một trong k láng giềng; đặc trưng rời rạc lấy giá trị phổ biến nhất trong k láng giềng.
    Chỉ nên áp dụng cho tập train."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    num_cols = [c for c in X.columns if c in NUMERIC_COLS]
    cat_cols = [c for c in X.columns if c not in NUMERIC_COLS]
    n_new = int(ratio * (y == 0).sum()) - int((y == 1).sum())
    if n_new <= 0:
        return X.reset_index(drop=True), y
    Xm = X[y == 1].reset_index(drop=True)
    n_min = len(Xm)
    k = min(k, n_min - 1)
    parts, sd_med = [], 1.0
    if num_cols:
        num = Xm[num_cols].to_numpy(dtype=float)
        sd = num.std(axis=0)
        sd[sd == 0] = 1.0
        sd_med = float(np.median(num.std(axis=0))) or 1.0
        parts.append(num)  # giữ thang gốc: chênh lệch số tính theo đơn vị gốc, như SMOTENC
    for c in cat_cols:
        vals = Xm[c].to_numpy()
        levels = np.unique(vals)
        parts.append((vals[:, None] == levels[None, :]).astype(float) * (sd_med / 2.0))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(np.hstack(parts))
    idx = nn.kneighbors(np.hstack(parts), return_distance=False)[:, 1:]  # bỏ chính nó
    base = rng.randint(0, n_min, n_new)
    pick = idx[base, rng.randint(0, k, n_new)]
    step = rng.rand(n_new, 1)
    out = pd.DataFrame(index=range(n_new), columns=X.columns)
    if num_cols:
        a, b = Xm[num_cols].to_numpy(dtype=float)[base], Xm[num_cols].to_numpy(dtype=float)[pick]
        out[num_cols] = a + step * (b - a)
    for c in cat_cols:
        col = Xm[c].to_numpy()
        levels = np.unique(col)
        neigh = col[idx[base]]  # (n_new, k)
        counts = (neigh[:, :, None] == levels[None, None, :]).sum(axis=1) + rng.rand(n_new, len(levels)) * 0.01  # hòa: chọn ngẫu nhiên
        out[c] = levels[counts.argmax(axis=1)]
    out = out.astype({c: float for c in num_cols} | {c: X[c].dtype for c in cat_cols})
    Xr = pd.concat([X.reset_index(drop=True), out[X.columns]], ignore_index=True)
    yr = np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
    return Xr, yr


_SMOTENC_BACKEND = {"name": None}


def smotenc_builtin(X: pd.DataFrame, y, ratio: float, k: int, seed: int):
    """SMOTENC cài đặt riêng (không cần imbalanced-learn), theo Chawla et al. (2002) như thư viện imblearn:
    - khoảng cách = Euclid trên biến định lượng + hình phạt cho mỗi biến rời rạc khác nhau (bằng bình phương độ lệch chuẩn trung vị của biến định lượng);
    - mẫu mới: biến định lượng nội suy giữa mẫu gốc và một láng giềng ngẫu nhiên; biến rời rạc lấy giá trị phổ biến nhất trong k láng giềng.
    Chỉ sinh thêm mẫu lớp 1 sao cho n_lớp1 = ratio * n_lớp0."""
    from sklearn.neighbors import NearestNeighbors
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    cols = list(X.columns)
    cat_cols = [c for c in cols if c not in NUMERIC_COLS]
    num_cols = [c for c in cols if c in NUMERIC_COLS]
    Xm = X[y == 1].reset_index(drop=True)
    n_min, n_maj = len(Xm), int((y == 0).sum())
    n_new = int(ratio * n_maj) - n_min
    if n_new <= 0 or n_min <= 1:
        return X.reset_index(drop=True), y
    k = min(k, n_min - 1)
    num = Xm[num_cols].to_numpy(dtype=float) if num_cols else np.zeros((n_min, 0))
    med = float(np.median(num.std(axis=0))) if num.shape[1] else 1.0
    blocks = [num]
    for c in cat_cols:   # one-hot nhân hệ số để mỗi biến rời rạc khác nhau cộng đúng med^2 vào bình phương khoảng cách
        lv = np.unique(X[c].to_numpy())
        blocks.append((Xm[c].to_numpy()[:, None] == lv[None, :]).astype(float) * (med / np.sqrt(2.0)))
    emb = np.hstack(blocks)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    nbr = nn.kneighbors(emb, return_distance=False)[:, 1:]          # bỏ chính nó
    base = rng.randint(0, n_min, n_new)
    pick = nbr[base, rng.randint(0, k, n_new)]
    new = pd.DataFrame(index=range(n_new), columns=cols)
    gap = rng.rand(n_new, 1)
    if num_cols:
        nv = num[base] + gap * (num[pick] - num[base])
        for j, c in enumerate(num_cols):
            new[c] = nv[:, j]
    for c in cat_cols:
        vals = Xm[c].to_numpy()[nbr[base]]                           # (n_new, k) giá trị của k láng giềng
        lv = np.unique(X[c].to_numpy())
        counts = (vals[:, :, None] == lv[None, None, :]).sum(axis=1)
        new[c] = lv[counts.argmax(axis=1)]
    new = new[cols].astype({c: X[c].dtype for c in cat_cols})
    X_out = pd.concat([X.reset_index(drop=True), new], ignore_index=True)
    return X_out, np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


class SmotencWrapper:
    """Huấn luyện mô hình trên dữ liệu đã được SMOTENC (chỉ áp dụng cho tập train; validation/test giữ nguyên).
    Dùng imbalanced-learn nếu import được; nếu không (thiếu thư viện hoặc bị chặn, ví dụ chính sách Application Control của Windows)
    thì dùng bản cài đặt riêng smotenc_builtin."""

    def __init__(self, base, seed: int, ratio: float = SMOTE_RATIO):
        self.base, self.seed, self.ratio = base, seed, ratio

    def fit(self, X, y):
        if os.environ.get("DSS_SMOTENC_BACKEND") != "builtin" and _SMOTENC_BACKEND["name"] != "builtin":
            try:
                from imblearn.over_sampling import SMOTENC
                if _SMOTENC_BACKEND["name"] is None:
                    _SMOTENC_BACKEND["name"] = "imblearn"
                    print("SMOTENC: dùng imbalanced-learn")
                cat_idx = [i for i, c in enumerate(X.columns) if c not in NUMERIC_COLS] if hasattr(X, "columns") else CAT_IDX
                sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=self.ratio, k_neighbors=5, random_state=self.seed)
                Xr, yr = sm.fit_resample(X, y)
                self.base.fit(Xr, yr)
                return self
            except Exception as e:   # ImportError, OSError (DLL bị chặn), ...
                if _SMOTENC_BACKEND["name"] == "imblearn":
                    raise
                _SMOTENC_BACKEND["name"] = "builtin"
                print(f"SMOTENC: không dùng được imbalanced-learn ({type(e).__name__}: {str(e)[:90]}) -> dùng bản cài đặt riêng smotenc_builtin")
        if _SMOTENC_BACKEND["name"] is None:
            _SMOTENC_BACKEND["name"] = "builtin"
            print("SMOTENC: dùng bản cài đặt riêng smotenc_builtin")
        Xr, yr = smotenc_builtin(X, y, self.ratio, 5, self.seed)
        self.base.fit(Xr, yr)
        return self

    def predict_proba(self, X):
        return self.base.predict_proba(X)


def base_estimator(name: str, seed: int, class_weight: bool = False, pos_weight: float = 1.0, n_jobs: int = -1):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced" if class_weight else None))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", n_jobs=n_jobs,
                                      random_state=seed, class_weight="balanced_subsample" if class_weight else None)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, tree_method="hist", eval_metric="logloss",
                             n_jobs=n_jobs, random_state=seed, scale_pos_weight=pos_weight if class_weight else 1.0, verbosity=0)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample_freq=1, subsample=0.8,
                              random_state=seed, n_jobs=n_jobs, verbose=-1, scale_pos_weight=pos_weight if class_weight else 1.0)
    if name == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(random_state=seed, outer_bags=4, interactions=5, n_jobs=n_jobs if n_jobs != -1 else -2)
    raise ValueError(name)


# Không gian tìm kiếm siêu tham số (RandomizedSearchCV). Tên tham số của Logistic Regression có tiền tố của pipeline.
def param_space(name: str) -> dict:
    from scipy.stats import loguniform, randint, uniform
    if name == "Logistic Regression":
        return {"logisticregression__C": loguniform(1e-3, 1e2)}
    if name == "Random Forest":
        return {"n_estimators": randint(150, 450), "max_depth": [8, 12, 16, 24, None], "min_samples_leaf": randint(3, 60),
                "max_features": ["sqrt", 0.3, 0.5], "max_samples": [0.5, 0.8, None]}
    if name == "XGBoost":
        return {"n_estimators": randint(150, 600), "learning_rate": loguniform(0.02, 0.2), "max_depth": randint(3, 9),
                "min_child_weight": randint(1, 30), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                "reg_lambda": loguniform(0.5, 20)}
    if name == "LightGBM":
        return {"n_estimators": randint(150, 600), "learning_rate": loguniform(0.02, 0.2), "num_leaves": randint(15, 127),
                "min_child_samples": randint(20, 200), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                "reg_lambda": loguniform(0.5, 20)}
    if name == "EBM":
        return {"learning_rate": [0.01, 0.03, 0.05], "max_leaves": [2, 3, 4], "interactions": [0, 5, 10]}
    raise ValueError(name)


def build_estimator(name: str, params: dict | None, seed: int, strategy: str, y_train, n_jobs: int = -1):
    y_train = np.asarray(y_train)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    est = base_estimator(name, seed, class_weight=(strategy == "class_weight"), pos_weight=pos_weight, n_jobs=n_jobs)
    if params:
        est.set_params(**params)
    return SmotencWrapper(est, seed) if strategy == "smotenc" else est


def tune_model(name: str, split: dict, seed: int, n_iter: int, max_rows: int, cv_folds: int = 3, n_jobs: int = -1) -> dict:
    """RandomizedSearchCV trên tập train (cross-validation theo nhóm hồ sơ, tiêu chí PR-AUC). Không dùng validation/test."""
    X, y, g = split["train"]
    if len(y) > max_rows:
        rng = np.random.RandomState(seed)
        keep = np.sort(rng.choice(len(y), max_rows, replace=False))
        X, y, g = X.iloc[keep], y.iloc[keep], g[keep]
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    est = base_estimator(name, seed, n_jobs=n_jobs)
    search = RandomizedSearchCV(est, param_space(name), n_iter=n_iter, scoring="average_precision",
                                cv=list(cv.split(X, y, g)), random_state=seed, n_jobs=1, refit=False, error_score="raise")
    t0 = time.time()
    search.fit(X, y)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    print(f"   {name}: PR-AUC CV tốt nhất = {search.best_score_:.4f} ({time.time() - t0:.0f}s) | {best}")
    return {"params": best, "cv_pr_auc": float(search.best_score_), "n_iter": n_iter, "tune_rows": int(len(y)), "seed": seed}


# ----------------------------------------------------------------------------------------------
# Chỉ số
# ----------------------------------------------------------------------------------------------
def ece_score(y, p, bins: int = 10) -> float:
    """Expected Calibration Error với các bin theo phân vị."""
    y, p = np.asarray(y), np.asarray(p)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def prob_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"Brier": brier_score_loss(y, p), "Log Loss": log_loss(y, p, labels=[0, 1]), "ECE": ece_score(y, p),
            "XS TB - tỷ lệ thật": float(p.mean() - y.mean())}


def rank_metrics(y, p) -> dict:
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)}


def threshold_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"Ngưỡng": float(thr), "Precision": prec, "Recall": rec, "Specificity": spec, "F1": f1,
            "Balanced Accuracy": (rec + spec) / 2, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Tỷ lệ cảnh báo (%)": 100 * pred.mean()}


def thr_f1max(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.argmax(f1))])


def thr_recall(y, p, target: float = 0.80) -> float:
    """Ngưỡng cao nhất vẫn đạt Recall >= target (tức Precision lớn nhất thỏa điều kiện Recall)."""
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(rc[:-1] >= target)[0]
    return float(th[ok[-1]]) if len(ok) else float(th[0])


def thr_cost(y, p, c_fn: float, c_fp: float, grid: int = 2001) -> float:
    """Ngưỡng làm tối thiểu chi phí kỳ vọng c_fn*FN + c_fp*FP trên tập (validation)."""
    y, p = np.asarray(y), np.asarray(p)
    cand = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    fn = np.searchsorted(pos, cand, side="left")           # dương tính thật có p < ngưỡng
    fp = len(neg) - np.searchsorted(neg, cand, side="left")  # âm tính thật có p >= ngưỡng
    return float(cand[int(np.argmin(c_fn * fn + c_fp * fp))])


# ----------------------------------------------------------------------------------------------
# Hiệu chuẩn xác suất
# ----------------------------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """method: 'none' | 'sigmoid' (Platt scaling) | 'isotonic'."""

    def __init__(self, method: str):
        self.method = method

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "sigmoid":
            return self.m_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.m_.predict(p)
        return p


CAL_METHODS = ["none", "sigmoid", "isotonic"]
CAL_LABEL = {"none": "Chưa hiệu chuẩn", "sigmoid": "Platt (sigmoid)", "isotonic": "Isotonic"}


def crossfit_calibrate(p, y, groups, method: str, seed: int, k: int = 5) -> np.ndarray:
    """Xác suất đã hiệu chuẩn theo kiểu cross-fitting: mỗi fold được hiệu chuẩn bởi bộ hiệu chuẩn học từ các fold còn lại."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    if method == "none":
        return p
    out = np.empty_like(p)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in cv.split(p, y, groups):
        out[te] = Calibrator(method).fit(p[tr], y[tr]).transform(p[te])
    return out


# ----------------------------------------------------------------------------------------------
# Thống kê
# ----------------------------------------------------------------------------------------------
def mean_sd_ci(values) -> tuple[float, float, float, float]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return m, sd, m - h, m + h


def fmt_ci(values, digits: int = 4) -> str:
    m, sd, lo, hi = mean_sd_ci(values)
    return f"{m:.{digits}f} ± {sd:.{digits}f} [{lo:.{digits}f}; {hi:.{digits}f}]"


def paired_bootstrap_diff(y, p_a, p_b, metric=average_precision_score, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap ghép cặp theo bản ghi: chênh lệch metric(A) - metric(B) trên cùng tập đánh giá."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.RandomState(seed)
    n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.randint(0, n, n)
        d[b] = metric(y[i], p_a[i]) - metric(y[i], p_b[i])
    obs = metric(y, p_a) - metric(y, p_b)
    p_two = float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 2.0 / (n_boot + 1))))
    return {"Chênh lệch": float(obs), "CI95 dưới": float(np.percentile(d, 2.5)), "CI95 trên": float(np.percentile(d, 97.5)), "p (bootstrap, 2 phía)": p_two}


# ----------------------------------------------------------------------------------------------
# Quy trình thực nghiệm
# ----------------------------------------------------------------------------------------------
def tune_all(models, cfg: dict, cache_path: Path | None = None) -> dict:
    """Tinh chỉnh từng mô hình một lần trên train của (kịch bản B, seed đầu tiên); lưu cache JSON để chạy lại không phải tinh chỉnh."""
    cache_path = cache_path or (MODEL_DIR / "best_params.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    seed0 = cfg["seeds"][0]
    split = get_split("profile", seed0, cfg.get("max_train_rows"))
    for name in models:
        key = f"{name}|n_iter={cfg['n_iter']}|rows={cfg['tune_rows']}"
        if key in cache:
            print(f"   {name}: dùng kết quả tinh chỉnh đã lưu.")
            continue
        n_iter = min(cfg["n_iter"], 6) if name == "EBM" else cfg["n_iter"]
        cache[key] = tune_model(name, split, seed0, n_iter, cfg["tune_rows"], n_jobs=cfg.get("n_jobs", -1))
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return {name: cache[f"{name}|n_iter={cfg['n_iter']}|rows={cfg['tune_rows']}"]["params"] for name in models}


def run_stage_a(models, params_by_model: dict, cfg: dict, out_name: str = "2_stageA_validation_runs") -> pd.DataFrame:
    """Giai đoạn A: so sánh mô hình x chiến lược mất cân bằng x seed trên kịch bản B, CHỈ ĐÁNH GIÁ TRÊN VALIDATION.
    Có cơ chế chạy tiếp: các tổ hợp đã có trong file kết quả sẽ được bỏ qua."""
    path = TABLES / f"{out_name}.csv"
    done = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    have = {(r["Mô hình"], r["Chiến lược"], int(r["Seed"])) for r in rows}
    for seed in cfg["seeds"]:
        split = get_split("profile", seed, cfg.get("max_train_rows"))
        Xtr, ytr, _ = split["train"]
        Xva, yva, _ = split["validation"]
        for name in models:
            for strat in supported_strategies(name):
                if (name, strat, seed) in have:
                    continue
                t0 = time.time()
                est = build_estimator(name, params_by_model.get(name), seed, strat, ytr, cfg.get("n_jobs", -1)).fit(Xtr, ytr)
                p = est.predict_proba(Xva)[:, 1]
                row = {"Mô hình": name, "Chiến lược": strat, "Seed": seed, "Số bản ghi train": len(ytr), **rank_metrics(yva, p), **prob_metrics(yva, p)}
                t1, t2 = thr_f1max(yva, p), thr_recall(yva, p, 0.80)
                for tag, t in [("F1max", t1), ("Recall80", t2)]:
                    m = threshold_metrics(yva, p, t)
                    for k in ["Precision", "Recall", "Specificity", "F1", "Balanced Accuracy", "FN", "FP"]:
                        row[f"{k} @{tag}"] = m[k]
                row["Thời gian (s)"] = round(time.time() - t0, 1)
                rows.append(row)
                pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
                print(f"   seed {seed} | {name:20s} | {strat:12s} | PR-AUC {row['PR-AUC']:.4f} | ROC-AUC {row['ROC-AUC']:.4f} | {row['Thời gian (s)']}s")
    return pd.DataFrame(rows)


def select_strategies(stage_a: pd.DataFrame) -> pd.DataFrame:
    """Với mỗi mô hình chọn chiến lược mất cân bằng có PR-AUC trung bình trên validation lớn nhất (không dùng test)."""
    g = stage_a.groupby(["Mô hình", "Chiến lược"])["PR-AUC"].mean().reset_index()
    best = g.loc[g.groupby("Mô hình")["PR-AUC"].idxmax()].reset_index(drop=True)
    return best.rename(columns={"PR-AUC": "PR-AUC validation (TB)"})


def evaluate_final(choices: dict, params_by_model: dict, scenarios, cfg: dict, save_models: bool = True):
    """Giai đoạn B: huấn luyện lại cấu hình đã chọn trên train, hiệu chuẩn và chọn ngưỡng bằng validation, đánh giá trên TEST.
    choices: {tên mô hình: chiến lược}. Trả về (bảng chỉ số xác suất, bảng ngưỡng, bảng chi phí)."""
    prob_rows, thr_rows, cost_rows = [], [], []
    ratios = cfg["cost_ratios"]
    for scen in scenarios:
        for seed in cfg["seeds"]:
            split = get_split(scen, seed, cfg.get("max_train_rows"))
            Xtr, ytr, _ = split["train"]
            Xva, yva, gva = split["validation"]
            Xte, yte, _ = split["test"]
            for name, strat in choices.items():
                t0 = time.time()
                est = build_estimator(name, params_by_model.get(name), seed, strat, ytr, cfg.get("n_jobs", -1)).fit(Xtr, ytr)
                p_va, p_te = est.predict_proba(Xva)[:, 1], est.predict_proba(Xte)[:, 1]
                # hiệu chuẩn: đánh giá trên validation bằng cross-fitting; bộ hiệu chuẩn cuối học trên toàn validation, áp dụng cho test
                cal_va, cal_te, cals = {}, {}, {}
                for meth in CAL_METHODS:
                    cal_va[meth] = crossfit_calibrate(p_va, yva, gva, meth, seed)
                    cals[meth] = Calibrator(meth).fit(p_va, yva)
                    cal_te[meth] = cals[meth].transform(p_te)
                    for split_name, yy, pp in [("validation (cross-fit)", yva, cal_va[meth]), ("test", yte, cal_te[meth])]:
                        prob_rows.append({"Kịch bản": scen, "Mô hình": name, "Chiến lược": strat, "Seed": seed, "Hiệu chuẩn": meth,
                                          "Tập": split_name, **rank_metrics(yy, pp), **prob_metrics(yy, pp)})
                # phương pháp hiệu chuẩn được chọn: Brier nhỏ nhất trên validation (cross-fit)
                best_cal = min(CAL_METHODS, key=lambda m: brier_score_loss(yva, np.clip(cal_va[m], 1e-6, 1 - 1e-6)))
                pv, pt = cal_va[best_cal], cal_te[best_cal]
                ops = {"F1 lớn nhất": thr_f1max(yva, pv), "Recall >= 80%": thr_recall(yva, pv, 0.80), "Recall >= 90%": thr_recall(yva, pv, 0.90)}
                for r in ratios:
                    ops[f"Chi phí FN:FP = {r}:1"] = thr_cost(yva, pv, r, 1.0)
                for op, thr in ops.items():
                    m_te = threshold_metrics(yte, pt, thr)
                    m_va = threshold_metrics(yva, pv, thr)
                    thr_rows.append({"Kịch bản": scen, "Mô hình": name, "Chiến lược": strat, "Seed": seed, "Hiệu chuẩn": best_cal,
                                     "Điểm vận hành": op, **m_te, "Recall (validation)": m_va["Recall"], "Precision (validation)": m_va["Precision"]})
                    if op.startswith("Chi phí"):
                        r = float(op.split("=")[1].split(":")[0])
                        cost_rows.append({"Kịch bản": scen, "Mô hình": name, "Seed": seed, "Tỷ lệ chi phí FN:FP": r, "Ngưỡng": thr,
                                          "Chi phí TB / 1000 người (test)": 1000 * (r * m_te["FN"] + m_te["FP"]) / len(yte),
                                          "Chi phí nếu không cảnh báo ai": 1000 * r * yte.sum() / len(yte),
                                          "Chi phí nếu cảnh báo tất cả": 1000 * (len(yte) - yte.sum()) / len(yte),
                                          "FN": m_te["FN"], "FP": m_te["FP"], "Recall": m_te["Recall"], "Precision": m_te["Precision"]})
                if seed == cfg["seeds"][0]:
                    np.savez_compressed(PRED_DIR / f"{scen}_{name.replace(' ', '_')}_seed{seed}.npz", y_test=yte.to_numpy(), p_raw=p_te,
                                        p_sigmoid=cal_te["sigmoid"], p_isotonic=cal_te["isotonic"], p_best=pt, best_cal=best_cal,
                                        y_val=yva.to_numpy(), p_val_raw=p_va, p_val_cf_best=pv)
                    if save_models and scen == "profile":
                        import joblib
                        joblib.dump({"model": est, "calibrator": cals[best_cal], "calibration": best_cal, "thresholds": ops, "features": FEATURES_21,
                                     "model_name": name, "strategy": strat, "params": params_by_model.get(name), "seed": seed,
                                     "target_meaning": "Xác suất thuộc nhóm có tiền sử được chẩn đoán tiền tiểu đường/tiểu đường theo dữ liệu khảo sát (không phải xác suất mắc bệnh trong tương lai)"},
                                    MODEL_DIR / f"{name.replace(' ', '_')}_bundle.joblib", compress=3)
                print(f"   [{scen}] seed {seed} | {name:20s} | {strat:12s} | hiệu chuẩn: {best_cal:8s} | {time.time() - t0:.0f}s")
    return pd.DataFrame(prob_rows), pd.DataFrame(thr_rows), pd.DataFrame(cost_rows)


# ----------------------------------------------------------------------------------------------
# Tiện ích tổng hợp bảng
# ----------------------------------------------------------------------------------------------
def summarize(df: pd.DataFrame, by, cols, digits: int = 4) -> pd.DataFrame:
    """Gom theo `by`, mỗi cột thành 'trung bình ± độ lệch chuẩn [CI95 dưới; trên]' (CI theo phân phối t qua các seed)."""
    by = [by] if isinstance(by, str) else list(by)
    out = df.groupby(by, sort=False).agg(**{"Số seed": ("Seed", "nunique")}).reset_index()
    for c in cols:
        vals = df.groupby(by, sort=False)[c].apply(lambda s: fmt_ci(s, digits)).reset_index(drop=True)
        out[c] = vals
    return out


def holm_adjust(pvals) -> np.ndarray:
    """Hiệu chỉnh Holm cho nhiều phép so sánh."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------------------------
# Thí nghiệm bỏ dần biến yếu (chỉ dùng train/validation)
# ----------------------------------------------------------------------------------------------
def permutation_importance_pr(est, X: pd.DataFrame, y, n_repeats: int = 3, seed: int = 0, max_rows: int = 20000) -> pd.DataFrame:
    """Độ giảm PR-AUC khi xáo trộn từng đặc trưng (tính trên tập được đưa vào, ở đây là validation)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.RandomState(seed)
    if len(y) > max_rows:
        keep = rng.choice(len(y), max_rows, replace=False)
        X, y = X.iloc[keep].reset_index(drop=True), np.asarray(y)[keep]
    y = np.asarray(y)
    base = average_precision_score(y, est.predict_proba(X)[:, 1])
    rows = []
    for f in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - average_precision_score(y, est.predict_proba(Xp)[:, 1]))
        rows.append({"Đặc trưng": f, "Giảm PR-AUC khi xáo trộn": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("Giảm PR-AUC khi xáo trộn", ascending=False).reset_index(drop=True)


def run_feature_ablation(model_name: str, strategy: str, params: dict | None, cfg: dict,
                         drop_counts=(0, 1, 2, 3, 5, 8, 10), extra_sets: dict | None = None):
    """Xếp hạng đặc trưng bằng permutation importance trên VALIDATION của seed đầu tiên, rồi huấn luyện lại mô hình
    sau khi bỏ dần k đặc trưng yếu nhất (và các tập tùy chọn như bỏ Education). Chỉ đánh giá trên validation, qua mọi seed."""
    seed0 = cfg["seeds"][0]
    sp0 = get_split("profile", seed0, cfg.get("max_train_rows"))
    Xtr, ytr, _ = sp0["train"]
    Xva, yva, _ = sp0["validation"]
    est0 = build_estimator(model_name, params, seed0, strategy, ytr, cfg.get("n_jobs", -1)).fit(Xtr, ytr)
    imp = permutation_importance_pr(est0, Xva, yva, seed=seed0)
    order_weak_first = list(imp["Đặc trưng"].iloc[::-1])
    variants = {("Đủ 21 biến" if k == 0 else f"Bỏ {k} biến yếu nhất"): order_weak_first[:k] for k in drop_counts}
    for name, cols in (extra_sets or {}).items():
        variants[name] = list(cols)
    rows = []
    for seed in cfg["seeds"]:
        split = get_split("profile", seed, cfg.get("max_train_rows"))
        Xt, yt, _ = split["train"]
        Xv, yv, _ = split["validation"]
        for vname, dropped in variants.items():
            feats = [f for f in FEATURES_21 if f not in dropped]
            est = build_estimator(model_name, params, seed, strategy, yt, cfg.get("n_jobs", -1)).fit(Xt[feats], yt)
            p = est.predict_proba(Xv[feats])[:, 1]
            m = threshold_metrics(yv, p, thr_recall(yv, p, 0.80))
            rows.append({"Phương án": vname, "Số biến bỏ": len(dropped), "Số biến còn lại": len(feats), "Seed": seed,
                         **rank_metrics(yv, p), "Brier": brier_score_loss(yv, np.clip(p, 1e-6, 1 - 1e-6)),
                         "Precision @Recall80": m["Precision"], "Recall @Recall80": m["Recall"]})
        print(f"   seed {seed}: xong {len(variants)} phương án")
    return imp, pd.DataFrame(rows), variants
