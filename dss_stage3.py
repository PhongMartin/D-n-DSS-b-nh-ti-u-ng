"""Giai đoạn 3 – Hệ hỗ trợ quyết định (DSS) đánh giá nguy cơ tiền tiểu đường/tiểu đường (CDC BRFSS 2015).

Luồng hệ thống:  Dữ liệu người dùng (21 biến) → LightGBM → xác suất (đã/chưa hiệu chuẩn, tùy kết quả so sánh trên validation)
→ phân mức nguy cơ → bộ quy tắc (rules/decision_rules.json) → mức ưu tiên + hành động tham khảo → giải thích (SHAP).

Nguyên tắc chống rò rỉ (giữ nguyên từ Giai đoạn 2):
- Mọi lựa chọn thiết kế (hiệu chuẩn, ngưỡng, sơ đồ phân mức) được quyết định bằng validation; test chỉ dùng để báo cáo cuối.
- Mỗi seed có một mô hình riêng, huấn luyện trên train của chính seed đó (không đánh giá mô hình seed này trên test seed khác).

Lưu ý diễn giải: nhãn là thông tin TỰ KHAI về tiền sử được chẩn đoán (khảo sát 2015). Xác suất không phải xác suất mắc bệnh trong tương lai;
khuyến nghị chỉ mang tính tham khảo, không thay thế đánh giá của nhân viên y tế.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import dss_utils as U

FEATURES = U.FEATURES_21
SEEDS = [42, 43, 44, 45, 46]
FOLDER, TABLES, FIGURES, MODEL_DIR = U.FOLDER, U.TABLES, U.FIGURES, U.MODEL_DIR
P3_DIR = U.RESULTS / "phase3"
RULES_DIR = FOLDER / "rules"
for _d in (P3_DIR, RULES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
SYSTEM_PATH = MODEL_DIR / "Phase3_system.joblib"
RULES_PATH = RULES_DIR / "decision_rules.json"
TESTCASE_PATH = RULES_DIR / "test_cases.csv"
COST_RATIOS = [2, 5, 10]          # tỉ lệ chi phí FN:FP – GIẢ ĐỊNH để phân tích độ nhạy, không phải số liệu thực tế
CAL_MIN_GAIN = 0.0001             # quy tắc chọn hiệu chuẩn (đặt trước): chỉ chọn Platt/Isotonic nếu giảm Brier ở mọi seed và trung bình ≥ ngưỡng này
TIER_NAMES = {"low": "Thấp", "medium": "Trung bình", "high": "Cao"}
TIER_PRIORITY = {"low": 1, "medium": 2, "high": 3}
PRIORITY_NAMES = {1: "Theo dõi", 2: "Cần lưu ý", 3: "Ưu tiên cao"}

# ----------------------------------------------------------------------------------------------
# Mô tả biến (để giao diện không bắt người dùng nhập mã số)
# ----------------------------------------------------------------------------------------------
AGE_LABELS = {1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59",
              9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80 trở lên"}
GENHLTH_LABELS = {1: "Rất tốt", 2: "Tốt", 3: "Khá", 4: "Yếu", 5: "Kém"}
EDUCATION_LABELS = {1: "Chưa đi học/mầm non", 2: "Tiểu học", 3: "Một phần THPT", 4: "Tốt nghiệp THPT",
                    5: "Một phần đại học/cao đẳng", 6: "Tốt nghiệp đại học trở lên"}
INCOME_LABELS = {1: "< 10.000 USD/năm", 2: "10.000–15.000", 3: "15.000–20.000", 4: "20.000–25.000",
                 5: "25.000–35.000", 6: "35.000–50.000", 7: "50.000–75.000", 8: "≥ 75.000"}
SEX_LABELS = {0: "Nữ", 1: "Nam"}
YESNO = {0: "Không", 1: "Có"}
FEATURE_VN = {
    "HighBP": "Huyết áp cao", "HighChol": "Cholesterol cao", "CholCheck": "Đã kiểm tra cholesterol (5 năm qua)",
    "BMI": "BMI", "Smoker": "Đã từng hút ≥100 điếu thuốc", "Stroke": "Tiền sử đột quỵ",
    "HeartDiseaseorAttack": "Bệnh tim mạch vành/nhồi máu", "PhysActivity": "Có vận động thể chất (30 ngày qua)",
    "Fruits": "Ăn trái cây hằng ngày", "Veggies": "Ăn rau hằng ngày", "HvyAlcoholConsump": "Uống rượu nhiều",
    "AnyHealthcare": "Có bảo hiểm/chăm sóc y tế", "NoDocbcCost": "Không đi khám vì chi phí", "GenHlth": "Sức khỏe tự đánh giá",
    "MentHlth": "Số ngày sức khỏe tinh thần kém (30 ngày)", "PhysHlth": "Số ngày sức khỏe thể chất kém (30 ngày)",
    "DiffWalk": "Khó đi bộ/leo cầu thang", "Sex": "Giới tính", "Age": "Nhóm tuổi", "Education": "Học vấn", "Income": "Thu nhập",
}
VALUE_LABELS = {"Age": AGE_LABELS, "GenHlth": GENHLTH_LABELS, "Education": EDUCATION_LABELS, "Income": INCOME_LABELS, "Sex": SEX_LABELS}
BINARY_FEATURES = [f for f in FEATURES if f not in ("BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income", "Sex")]


def _save_fig(fig, name: str):
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"-> Đã lưu biểu đồ: {path}")
    return path


def _save_table(df: pd.DataFrame, name: str):
    U.save_table(df, name[:-4] if name.endswith(".csv") else name)
    return df


# ----------------------------------------------------------------------------------------------
# 3.0  Huấn luyện/nạp mô hình theo từng seed, chọn hiệu chuẩn bằng validation, chốt ngưỡng
# ----------------------------------------------------------------------------------------------
def phase2_config() -> dict:
    """Lấy cấu hình LightGBM đã chốt ở Giai đoạn 2 (siêu tham số + chiến lược mất cân bằng) từ bundle đã lưu."""
    path = MODEL_DIR / "LightGBM_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError("Chưa có models/LightGBM_bundle.joblib – hãy chạy xong Giai đoạn 2 (mục 2.4) trước.")
    b = joblib.load(path)
    return {"params": b["params"], "strategy": b["strategy"], "phase2_calibration": b["calibration"]}


def fit_seed(seed: int, cfg: dict, force: bool = False) -> dict:
    """Huấn luyện LightGBM (cấu hình Giai đoạn 2) trên train của `seed`; lưu xác suất thô trên validation và test."""
    f = P3_DIR / f"lgbm_seed{seed}.joblib"
    if f.exists() and not force:
        return joblib.load(f)
    t0 = time.time()
    sp = U.get_split("profile", seed)
    Xtr, ytr, _ = sp["train"]
    Xva, yva, gva = sp["validation"]
    Xte, yte, _ = sp["test"]
    est = U.build_estimator("LightGBM", cfg["params"], seed, cfg["strategy"], ytr, -1).fit(Xtr, ytr)
    run = {"seed": seed, "model": est, "X_val": Xva, "y_val": yva.to_numpy(), "g_val": gva, "X_test": Xte, "y_test": yte.to_numpy(),
           "p_val_raw": est.predict_proba(Xva)[:, 1], "p_test_raw": est.predict_proba(Xte)[:, 1]}
    if seed == SEEDS[0]:
        run["X_train"] = Xtr.astype("int16")
    joblib.dump(run, f, compress=3)
    print(f"   seed {seed}: huấn luyện xong ({time.time() - t0:.0f}s) | PR-AUC test (thô) = {U.rank_metrics(yte, run['p_test_raw'])['PR-AUC']:.4f}")
    return run


def fit_all(cfg: dict, seeds=SEEDS, force: bool = False) -> list:
    return [fit_seed(s, cfg, force) for s in seeds]


def calibration_comparison(runs: list):
    """So sánh Chưa hiệu chuẩn / Platt / Isotonic trên VALIDATION (cross-fitting theo hồ sơ). Trả về (bảng chi tiết, bảng tóm tắt, lựa chọn)."""
    rows = []
    for r in runs:
        for m in U.CAL_METHODS:
            pv = U.crossfit_calibrate(r["p_val_raw"], r["y_val"], r["g_val"], m, r["seed"])
            rows.append({"Seed": r["seed"], "Hiệu chuẩn": m, **U.prob_metrics(r["y_val"], pv), **U.rank_metrics(r["y_val"], pv)})
    detail = pd.DataFrame(rows)
    piv = detail.pivot(index="Seed", columns="Hiệu chuẩn", values="Brier")
    choice, why = "none", "Không phương pháp nào giảm Brier ở mọi seed với mức giảm đủ lớn → giữ xác suất chưa hiệu chuẩn (đơn giản hơn)."
    for m in ("sigmoid", "isotonic"):
        gain = (piv["none"] - piv[m])
        if (gain > 0).all() and gain.mean() >= CAL_MIN_GAIN and (choice == "none" or gain.mean() > (piv["none"] - piv[choice]).mean()):
            choice, why = m, f"{U.CAL_LABEL[m]} giảm Brier ở cả {len(gain)} seed (trung bình {gain.mean():.5f})."
    summ = U.summarize(detail, "Hiệu chuẩn", ["Brier", "Log Loss", "ECE", "PR-AUC"])
    summ["Hiệu chuẩn"] = summ["Hiệu chuẩn"].map(U.CAL_LABEL)
    _save_table(detail, "3_0_calibration_validation_by_seed.csv")
    _save_table(summ, "3_0_calibration_validation_summary.csv")
    return detail, summ, choice, why


def operating_points(y_val, p_val) -> dict:
    ops = {"Ngưỡng mặc định 0,5": 0.5, "F1 lớn nhất": U.thr_f1max(y_val, p_val),
           "Recall >= 80%": U.thr_recall(y_val, p_val, 0.80), "Recall >= 90%": U.thr_recall(y_val, p_val, 0.90)}
    for r in COST_RATIOS:
        ops[f"Chi phí FN:FP = {r}:1"] = U.thr_cost(y_val, p_val, r, 1.0)
    return ops


def finalize(run: dict, calibration: str) -> dict:
    """Áp dụng hiệu chuẩn đã chọn; ngưỡng chỉ học trên validation (p_val: cross-fit nếu có hiệu chuẩn)."""
    cal = U.Calibrator(calibration).fit(run["p_val_raw"], run["y_val"])
    p_val = U.crossfit_calibrate(run["p_val_raw"], run["y_val"], run["g_val"], calibration, run["seed"])
    out = dict(run, calibrator=cal, calibration=calibration, p_val=p_val, p_test=cal.transform(run["p_test_raw"]))
    out["ops"] = operating_points(out["y_val"], p_val)
    return out


def plot_architecture():
    """Sơ đồ kiến trúc hệ hỗ trợ quyết định: đầu vào, đầu ra và nhiệm vụ từng thành phần."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")
    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=7.6, linespacing=1.35)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    box(0.2, 4.4, 2.6, 2.2, "1. Dữ liệu người dùng", "Vào: 21 biến khảo sát\n(nhóm tuổi, BMI, huyết áp...)\nRa: bản ghi đã kiểm tra\ngiá trị hợp lệ", "#E8F1FA")
    box(3.4, 4.4, 2.8, 2.2, "2. Mô hình học máy", "LightGBM (+ hiệu chuẩn nếu được\nchọn trên validation)\nVào: 21 biến   Ra: xác suất p\nChỉ ước lượng xác suất", "#E6F4EA")
    box(6.8, 4.4, 3.0, 2.2, "3. Phân mức nguy cơ", "Vào: p, ngưỡng (chọn trên validation)\nRa: mức Thấp/Trung bình/Cao\nPhương án: ngưỡng cố định hoặc\nTop-k theo năng lực xử lý", "#FFF4D6")
    box(10.4, 4.4, 3.4, 2.2, "4. Bộ quy tắc (Decision Engine)", "Vào: mức nguy cơ + 21 biến\nRa: quy tắc kích hoạt, mức ưu tiên,\nhành động tham khảo, nguồn\nĐộc lập với mô hình (JSON)", "#FDE7E7")
    box(3.4, 1.4, 2.8, 2.2, "5. Giải thích (SHAP)", "Vào: mô hình + hồ sơ\nRa: biến làm tăng/giảm đầu ra\n(log-odds thô), theo hồ sơ/nhóm\nCung cấp bằng chứng, không ra quyết định", "#EFE8F8")
    box(6.8, 1.4, 3.0, 2.2, "6. Phân tích phương án", "Vào: hồ sơ, thay đổi được phép\nRa: so sánh trước/sau (What-if),\nđánh đổi Recall–Precision–số hồ sơ\nkhông chứng minh nhân quả", "#E3F6F5")
    box(10.4, 1.4, 3.4, 2.2, "7. Kết quả cho người dùng", "Mức nguy cơ + mức ưu tiên + lý do\n+ hành động tham khảo + giới hạn\n(mức thấp ≠ chắc chắn không mắc)\nXuất CSV/Excel", "#F2F2F2")
    for a, b in [((2.85, 5.5), (3.35, 5.5)), ((6.25, 5.5), (6.75, 5.5)), ((9.85, 5.5), (10.35, 5.5))]:
        arrow(*a, *b)
    arrow(4.8, 4.35, 4.8, 3.65); arrow(8.3, 4.35, 8.3, 3.65); arrow(12.1, 4.35, 12.1, 3.65)
    arrow(6.25, 2.5, 6.75, 2.5); arrow(9.85, 2.5, 10.35, 2.5)
    ax.text(7, 0.55, "Nhãn dữ liệu là thông tin TỰ KHAI về tiền sử chẩn đoán (BRFSS 2015); hệ thống hỗ trợ cân nhắc quyết định, không thay thế nhân viên y tế.",
            ha="center", fontsize=8.5, style="italic")
    ax.set_title("Kiến trúc hệ hỗ trợ quyết định – Giai đoạn 3", fontsize=12, fontweight="bold")
    _save_fig(fig, "3_0_architecture.png")
    return fig


# ----------------------------------------------------------------------------------------------
# 3.1  Bảng so sánh các điểm vận hành + phân mức 3 tầng
# ----------------------------------------------------------------------------------------------
def _per1000(m: dict, n: int) -> dict:
    return {"Cảnh báo / 1.000": 1000 * (m["TP"] + m["FP"]) / n, "TP / 1.000": 1000 * m["TP"] / n,
            "FN / 1.000": 1000 * m["FN"] / n, "FP / 1.000": 1000 * m["FP"] / n}


def operating_table(fins: list):
    rows = []
    for r in fins:
        for op, thr in r["ops"].items():
            m = U.threshold_metrics(r["y_test"], r["p_test"], thr)
            rows.append({"Seed": r["seed"], "Điểm vận hành": op, **m, **_per1000(m, len(r["y_test"]))})
    detail = pd.DataFrame(rows)
    cols = ["Ngưỡng", "Recall", "Precision", "F1", "FN / 1.000", "FP / 1.000", "Cảnh báo / 1.000"]
    summ = U.summarize(detail, "Điểm vận hành", cols, 3)
    order = list(fins[0]["ops"].keys())
    summ = summ.set_index("Điểm vận hành").loc[order].reset_index()
    _save_table(detail, "3_1_operating_points_by_seed.csv")
    _save_table(summ, "3_1_operating_points_summary.csv")
    return detail, summ


def tier_schemes(r: dict) -> dict:
    o = r["ops"]
    return {"A (Recall ≥ 80% → F1 lớn nhất)": (o["Recall >= 80%"], o["F1 lớn nhất"]),
            "B (Recall ≥ 90% → Recall ≥ 80%)": (o["Recall >= 90%"], o["Recall >= 80%"])}


def assign_tier(p, t_low: float, t_high: float):
    p = np.asarray(p, dtype=float)
    return np.where(p >= t_high, "high", np.where(p >= t_low, "medium", "low"))


def tier_evaluation(fins: list, split: str):
    """split='validation' (dùng để CHỌN sơ đồ) hoặc 'test' (báo cáo cuối)."""
    key_y, key_p = ("y_val", "p_val") if split == "validation" else ("y_test", "p_test")
    rows = []
    for r in fins:
        y, p = r[key_y], r[key_p]
        for name, (t1, t2) in tier_schemes(r).items():
            tier = assign_tier(p, t1, t2)
            for t in ("low", "medium", "high"):
                m = tier == t
                rows.append({"Sơ đồ": name, "Tầng": TIER_NAMES[t], "Seed": r["seed"], "Tỷ lệ dân số (%)": 100 * m.mean(),
                             "Tỷ lệ lớp 1 trong tầng (%)": 100 * y[m].mean() if m.any() else np.nan,
                             "Tỷ lệ ca bệnh được bắt (%)": 100 * y[m].sum() / y.sum()})
    detail = pd.DataFrame(rows)
    summ = U.summarize(detail, ["Sơ đồ", "Tầng"], ["Tỷ lệ dân số (%)", "Tỷ lệ lớp 1 trong tầng (%)", "Tỷ lệ ca bệnh được bắt (%)"], 2)
    _save_table(summ, f"3_1_tiers_{split}_summary.csv")
    return detail, summ


def plot_tiers(detail_test: pd.DataFrame):
    g = detail_test.groupby(["Sơ đồ", "Tầng"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"Thấp": "#4CAF50", "Trung bình": "#F5A623", "Cao": "#D9534F"}
    for ax, col in zip(axes, ["Tỷ lệ dân số (%)", "Tỷ lệ lớp 1 trong tầng (%)", "Tỷ lệ ca bệnh được bắt (%)"]):
        pv = g.pivot(index="Sơ đồ", columns="Tầng", values=col)[["Thấp", "Trung bình", "Cao"]]
        pv.plot(kind="bar", ax=ax, color=[colors[c] for c in pv.columns], rot=0, width=0.75)
        ax.set_title(col, fontsize=11); ax.set_xlabel(""); ax.legend(title="Tầng", fontsize=8)
        ax.set_xticklabels([s.split(" ")[0] for s in pv.index])
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f", fontsize=8)
    fig.suptitle("Phân mức nguy cơ 3 tầng – tập test, trung bình 5 seed (A: Recall≥80% & F1 lớn nhất; B: Recall≥90% & Recall≥80%)", fontsize=11)
    _save_fig(fig, "3_1_tier_distribution.png")
    return fig


# ----------------------------------------------------------------------------------------------
# 3.2  Lựa chọn theo năng lực xử lý: Top-k so với ngưỡng cố định
# ----------------------------------------------------------------------------------------------
def _topk_metrics(y, p, k_abs: int) -> dict:
    order = np.argsort(-p, kind="mergesort")[:k_abs]
    tp = int(y[order].sum())
    pos = int(y.sum())
    return {"Precision@k": tp / max(k_abs, 1), "Recall@k": tp / pos, "FN còn lại / 1.000": 1000 * (pos - tp) / len(y),
            "FP / 1.000": 1000 * (k_abs - tp) / len(y), "Hồ sơ xử lý / 1.000": 1000 * k_abs / len(y)}


def topk_comparison(fins: list, ks=(100, 200, 300, 400)):
    """So sánh Top-k (khối lượng cố định) với ngưỡng cố định (khối lượng thay đổi theo seed/tập)."""
    ops = ("F1 lớn nhất", "Recall >= 80%", "Recall >= 90%")
    mean_alerts = {op: 1000 * np.mean([(U.threshold_metrics(r["y_test"], r["p_test"], r["ops"][op])["TP"] + U.threshold_metrics(r["y_test"], r["p_test"], r["ops"][op])["FP"]) / len(r["y_test"]) for r in fins]) for op in ops}
    rows = []
    for r in fins:
        y, p = r["y_test"], r["p_test"]
        n = len(y)
        for k in ks:
            rows.append({"Seed": r["seed"], "Phương án": f"Top-k: {k}/1.000", **_topk_metrics(y, p, round(k / 1000 * n))})
        for op in ops:
            m = U.threshold_metrics(y, p, r["ops"][op])
            rows.append({"Seed": r["seed"], "Phương án": f"Ngưỡng cố định: {op}", "Precision@k": m["Precision"], "Recall@k": m["Recall"],
                         "FN còn lại / 1.000": 1000 * m["FN"] / n, "FP / 1.000": 1000 * m["FP"] / n, "Hồ sơ xử lý / 1.000": 1000 * (m["TP"] + m["FP"]) / n})
            rows.append({"Seed": r["seed"], "Phương án": f"Top-k bằng số cảnh báo TB của '{op}' ({mean_alerts[op]:.0f}/1.000)", **_topk_metrics(y, p, round(mean_alerts[op] / 1000 * n))})
    detail = pd.DataFrame(rows)
    summ = U.summarize(detail, "Phương án", ["Hồ sơ xử lý / 1.000", "Precision@k", "Recall@k", "FN còn lại / 1.000", "FP / 1.000"], 3)
    order = list(dict.fromkeys(detail["Phương án"]))
    summ = summ.set_index("Phương án").loc[order].reset_index()
    _save_table(summ, "3_2_topk_vs_threshold.csv")
    return detail, summ


def plot_topk(detail: pd.DataFrame):
    g = detail.groupby("Phương án").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    tk = g[g.index.str.startswith("Top-k:")]
    ax.plot(tk["Hồ sơ xử lý / 1.000"], tk["Recall@k"], "o-", color="#1f77b4", label="Top-k (xếp hạng theo xác suất)")
    for idx, row in g[g.index.str.startswith("Ngưỡng cố định")].iterrows():
        ax.scatter(row["Hồ sơ xử lý / 1.000"], row["Recall@k"], s=70, marker="s", color="#d62728")
        ax.annotate(idx.replace("Ngưỡng cố định: ", ""), (row["Hồ sơ xử lý / 1.000"], row["Recall@k"]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.scatter([], [], marker="s", color="#d62728", label="Ngưỡng cố định (chọn trên validation)")
    ax.set_xlabel("Số hồ sơ được ưu tiên xử lý trên 1.000 người"); ax.set_ylabel("Recall (tỷ lệ ca bệnh được phát hiện)")
    ax.set_title("Năng lực xử lý và Recall – test, trung bình 5 seed\n(năng lực là kịch bản GIẢ ĐỊNH)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save_fig(fig, "3_2_topk_vs_threshold.png")
    return fig


# ----------------------------------------------------------------------------------------------
# 3.3  Phân tích bỏ sót (FN) theo nhóm, có khoảng tin cậy
# ----------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def group_columns(X: pd.DataFrame) -> dict:
    return {
        "Nhóm tuổi": pd.cut(X["Age"], [0, 4, 8, 13], labels=["18–39", "40–59", "60 trở lên"]).astype(str),
        "BMI": pd.cut(X["BMI"], [0, 24.999, 29.999, 200], labels=["< 25", "25–29,9", "≥ 30"]).astype(str),
        "Giới tính": X["Sex"].map(SEX_LABELS),
        "Thu nhập": pd.cut(X["Income"], [0, 3, 6, 8], labels=["Thấp (< 20.000 USD)", "Trung bình (20.000–50.000)", "Cao (≥ 50.000)"]).astype(str),
    }


def fn_by_group(fin: dict, ops=("Recall >= 90%", "Recall >= 80%", "F1 lớn nhất"), min_pos: int = 100):
    X, y, p = fin["X_test"], fin["y_test"], fin["p_test"]
    rows = []
    groups = group_columns(X)
    for op in ops:
        thr = fin["ops"][op]
        pred = p >= thr
        for gname, gv in groups.items():
            for level in pd.unique(gv):
                m = (gv == level).to_numpy()
                n, npos = int(m.sum()), int(y[m].sum())
                tp, fn, fp = int((pred[m] & (y[m] == 1)).sum()), int((~pred[m] & (y[m] == 1)).sum()), int((pred[m] & (y[m] == 0)).sum())
                lo, hi = wilson(fn, npos)
                plo, phi = wilson(tp, tp + fp)
                rows.append({"Điểm vận hành": op, "Nhóm": gname, "Mức": level, "Số hồ sơ": n, "Số lớp 1": npos,
                             "Recall": tp / npos if npos else np.nan, "FNR": fn / npos if npos else np.nan,
                             "FNR CI95 dưới": lo, "FNR CI95 trên": hi, "Precision": tp / (tp + fp) if tp + fp else np.nan,
                             "Precision CI95 dưới": plo, "Precision CI95 trên": phi, "Tỷ lệ cảnh báo (%)": 100 * pred[m].mean(),
                             "Ghi chú": "ít mẫu lớp 1 – diễn giải thận trọng" if npos < min_pos else ""})
    df = pd.DataFrame(rows)
    _save_table(df, "3_3_fn_by_group.csv")
    return df


def plot_fn_by_group(df: pd.DataFrame, op: str = "Recall >= 80%"):
    d = df[df["Điểm vận hành"] == op].copy()
    d["Nhãn"] = d["Nhóm"] + ": " + d["Mức"]
    d = d.sort_values(["Nhóm", "Mức"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(d))[::-1]
    ax.barh(y, 100 * d["FNR"], color="#D9534F", alpha=0.85)
    ax.errorbar(100 * d["FNR"], y, xerr=[100 * (d["FNR"] - d["FNR CI95 dưới"]), 100 * (d["FNR CI95 trên"] - d["FNR"])], fmt="none", ecolor="k", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{a} (n+={b})" for a, b in zip(d["Nhãn"], d["Số lớp 1"])], fontsize=8)
    ax.set_xlabel("Tỷ lệ bỏ sót FNR (%) và khoảng tin cậy 95% (Wilson)")
    ax.set_title(f"Tỷ lệ bỏ sót theo nhóm – điểm vận hành '{op}' (test, seed 42)", fontsize=11)
    _save_fig(fig, "3_3_fn_by_group.png")
    return fig


# ----------------------------------------------------------------------------------------------
# 3.4  SHAP (TreeSHAP chính xác của LightGBM: predict(pred_contrib=True)), thang log-odds của đầu ra THÔ của mô hình
# ----------------------------------------------------------------------------------------------
def contribs(model, X: pd.DataFrame):
    """Trả về (SHAP (n×21), giá trị nền (n,)) trên thang log-odds của mô hình thô (trước hiệu chuẩn)."""
    X = X[FEATURES]
    booster = getattr(model, "booster_", None)
    c = np.asarray(booster.predict(X, pred_contrib=True) if booster is not None else model.predict(X, pred_contrib=True))
    return c[:, :-1], c[:, -1]


def check_additivity(model, X: pd.DataFrame, n: int = 500, tol: float = 1e-3):
    """Kiểm tra: tổng SHAP + nền = log-odds của xác suất thô."""
    Xs = X.iloc[:n]
    sv, base = contribs(model, Xs)
    p = np.clip(model.predict_proba(Xs[FEATURES])[:, 1], 1e-9, 1 - 1e-9)
    err = float(np.max(np.abs(sv.sum(axis=1) + base - np.log(p / (1 - p)))))
    print(f"-> Kiểm tra cộng tính SHAP: sai số lớn nhất {err:.2e} (log-odds)")
    assert err < tol, "SHAP không cộng tính – kiểm tra phiên bản lightgbm"
    return err


def _expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def shap_analysis(fins: list, n_sample: int = 2000, seed: int = 0):
    """SHAP toàn cục (seed 42), theo nhóm, độ ổn định qua 5 mô hình, và kiểm tra biến HvyAlcoholConsump."""
    ref = fins[0]
    X = ref["X_test"].reset_index(drop=True)
    idx = np.sort(np.random.RandomState(seed).choice(len(X), min(n_sample, len(X)), replace=False))
    Xs = X.iloc[idx].reset_index(drop=True)
    sv_by_seed = {}
    for r in fins:
        sv_by_seed[r["seed"]], base = contribs(r["model"], Xs)
        if r["seed"] == ref["seed"]:
            base0 = base
    sv = sv_by_seed[ref["seed"]]
    imp = pd.DataFrame({"Đặc trưng": FEATURES, "mean(|SHAP|)": np.abs(sv).mean(axis=0)}).sort_values("mean(|SHAP|)", ascending=False).reset_index(drop=True)
    imp["Hạng"] = np.arange(1, len(imp) + 1)
    _save_table(imp, "3_4_shap_importance.csv")
    # ổn định: Spearman trên mean|SHAP| của 21 biến + trùng khớp top-5, tính trên CÙNG mẫu (ổn định của GIẢI THÍCH, không phải của hiệu năng)
    mat = pd.DataFrame({f"Seed {s}": np.abs(v).mean(axis=0) for s, v in sv_by_seed.items()}, index=FEATURES)
    sp = mat.corr(method="spearman")
    tops = {c: set(mat[c].sort_values(ascending=False).index[:5]) for c in mat}
    pairs = [(a, b) for i, a in enumerate(mat.columns) for b in list(mat.columns)[i + 1:]]
    stab = {"Spearman TB (các cặp seed)": float(np.mean([sp.loc[a, b] for a, b in pairs])),
            "Spearman nhỏ nhất": float(np.min([sp.loc[a, b] for a, b in pairs])),
            "Trùng top-5 TB (/5)": float(np.mean([len(tops[a] & tops[b]) for a, b in pairs]))}
    _save_table(sp.reset_index().rename(columns={"index": "Seed"}), "3_4_shap_stability.csv")
    # theo nhóm
    groups = group_columns(Xs)
    top8 = list(imp["Đặc trưng"][:8])
    cols = {}
    for gname, gv in groups.items():
        for level in sorted(pd.unique(gv)):
            m = (gv == level).to_numpy()
            if m.sum() >= 30:
                cols[f"{gname}: {level}"] = np.abs(sv[m]).mean(axis=0)
    sub = pd.DataFrame(cols, index=FEATURES).loc[top8]
    _save_table(sub.reset_index().rename(columns={"index": "Đặc trưng"}), "3_4_shap_by_group.csv")
    # HvyAlcoholConsump
    h = Xs["HvyAlcoholConsump"].to_numpy()
    j = FEATURES.index("HvyAlcoholConsump")
    full = U.get_data()[0]
    alc = pd.DataFrame([{"HvyAlcoholConsump": int(v), "Số hồ sơ (mẫu SHAP)": int((h == v).sum()),
                         "SHAP trung bình (log-odds)": float(sv[h == v, j].mean()) if (h == v).any() else np.nan,
                         "Tỷ lệ lớp 1 trong toàn bộ dữ liệu (%)": 100 * full.loc[full["HvyAlcoholConsump"] == v, U.TARGET].mean()} for v in (0, 1)])
    _save_table(alc, "3_4_hvyalcohol_check.csv")
    return {"Xs": Xs, "sv": sv, "base": base0, "imp": imp, "stability": stab, "stability_matrix": sp, "by_group": sub, "alcohol": alc, "idx": idx}


def plot_shap_global(res: dict, top: int = 15):
    imp, sv, Xs = res["imp"], res["sv"], res["Xs"]
    feats = list(imp["Đặc trưng"][:top])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    d = imp.head(top).iloc[::-1]
    ax.barh(d["Đặc trưng"], d["mean(|SHAP|)"], color="#3b6ea5")
    ax.set_xlabel("mean(|SHAP|) – log-odds của mô hình thô"); ax.set_title("Độ quan trọng toàn cục (SHAP) – LightGBM, seed 42", fontsize=11)
    _save_fig(fig, "3_4_shap_bar.png")
    fig2, ax2 = plt.subplots(figsize=(8, 6.5))
    rng = np.random.RandomState(1)
    for i, f in enumerate(feats[::-1]):
        j = FEATURES.index(f)
        v = Xs[f].to_numpy(dtype=float)
        rank = (pd.Series(v).rank(pct=True).to_numpy()) if np.unique(v).size > 2 else (v - v.min()) / max(np.ptp(v), 1e-9)
        sc = ax2.scatter(sv[:, j], i + rng.uniform(-0.3, 0.3, len(v)), c=rank, cmap="coolwarm", s=5, alpha=0.7)
    ax2.set_yticks(range(len(feats))); ax2.set_yticklabels(feats[::-1]); ax2.axvline(0, color="grey", lw=0.8)
    ax2.set_xlabel("Giá trị SHAP (log-odds của mô hình thô): >0 làm tăng, <0 làm giảm đầu ra")
    cb = fig2.colorbar(sc, ax=ax2); cb.set_label("Giá trị đặc trưng (thấp → cao)")
    ax2.set_title("Phân bố SHAP theo đặc trưng (Top 15)", fontsize=11)
    _save_fig(fig2, "3_4_shap_beeswarm.png")
    return fig, fig2


def plot_shap_groups(res: dict):
    import seaborn as sns
    sub = res["by_group"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"label": "mean(|SHAP|)"})
    ax.set_title("Độ quan trọng SHAP theo nhóm đối tượng (8 đặc trưng hàng đầu)", fontsize=11); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    _save_fig(fig, "3_4_shap_by_group.png")
    return fig


def pick_cases(fin: dict, res: dict, t_low: float, t_high: float) -> list:
    """4 hồ sơ minh họa từ mẫu SHAP: thấp, trung bình, cao, và một ca bỏ sót (lớp 1, p thấp, trẻ/BMI<25/không huyết áp cao)."""
    Xs, idx = res["Xs"], res["idx"]
    y, p = fin["y_test"][idx], fin["p_test"][idx]
    def closest(mask, target):
        c = np.where(mask)[0]
        return int(c[np.argmin(np.abs(p[c] - target))]) if len(c) else None
    cases = [("Nguy cơ thấp", closest(y == 0, t_low / 3)),
             ("Nguy cơ trung bình", closest(y == 0, (t_low + t_high) / 2)),
             ("Nguy cơ cao", closest(y == 1, max(0.5, t_high * 2))),
             ("Ca bị bỏ sót (lớp 1, p thấp)", closest((y == 1) & (p < t_low) & (Xs["BMI"].to_numpy() < 25) & (Xs["HighBP"].to_numpy() == 0), t_low * 0.6))]
    return [(n, i) for n, i in cases if i is not None]


def plot_waterfalls(res: dict, fin: dict, cases: list, t_low: float, t_high: float):
    Xs, sv, base = res["Xs"], res["sv"], res["base"]
    n = len(cases)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9)) if n > 2 else plt.subplots(1, n, figsize=(7 * n, 5))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, i) in zip(axes, cases):
        row = sv[i]
        order = np.argsort(-np.abs(row))[:8]
        rest = row.sum() - row[order].sum()
        labels = [f"{FEATURES[j]} = {Xs.iloc[i][FEATURES[j]]:g}" for j in order] + ["các biến còn lại"]
        vals = list(row[order]) + [rest]
        cum = base[i]
        pts = [cum]
        for k, (lab, v) in enumerate(zip(labels, vals)):
            ax.barh(len(labels) - 1 - k, v, left=cum, color="#D9534F" if v > 0 else "#3b82c4")
            ax.text(cum + v + (0.02 if v > 0 else -0.02), len(labels) - 1 - k, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
            cum += v
            pts.append(cum)
        ax.set_xlim(min(pts) - 0.35, max(pts) + 0.45)
        fx = base[i] + row.sum()
        p_cal = float(fin["calibrator"].transform(np.atleast_1d(_expit(fx)))[0])
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.axvline(base[i], color="grey", ls=":", lw=1); ax.axvline(fx, color="k", ls="--", lw=1)
        tier = TIER_NAMES[str(assign_tier([p_cal], t_low, t_high)[0])]
        ax.set_title(f"{name}\nNhãn thực = {int(fin['y_test'][res['idx'][i]])} | f(x) = {fx:.2f} (log-odds thô) → xác suất = {p_cal:.3f} → mức {tier}", fontsize=9)
        ax.set_xlabel(f"log-odds thô (nền E[f(X)] = {base[i]:.2f}, tương đương xác suất {float(_expit(base[i])):.3f} – KHÔNG phải tỷ lệ nền)", fontsize=7)
    for ax in axes[len(cases):]:
        ax.axis("off")
    fig.suptitle("Giải thích từng hồ sơ (SHAP) – đóng góp của từng biến tính trên thang log-odds của mô hình thô", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "3_4_waterfall_cases.png")
    return fig


# ----------------------------------------------------------------------------------------------
# 3.5  Bộ quy tắc hỗ trợ quyết định (Decision Engine) – độc lập với mô hình học máy
# ----------------------------------------------------------------------------------------------
SRC_ADA2 = "ADA. 2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes—2026. Diabetes Care 2026;49(Suppl 1):S27 (doi:10.2337/dc26-S002)"
SRC_ADA3 = "ADA. 3. Prevention or Delay of Diabetes and Associated Comorbidities: Standards of Care in Diabetes—2026. Diabetes Care 2026;49(Suppl 1)"
SRC_ADA5 = "ADA. 5. Facilitating Positive Health Behaviors and Well-being: Standards of Care in Diabetes—2026. Diabetes Care 2026;49(Suppl 1)"
SRC_ADA10 = "ADA. 10. Cardiovascular Disease and Risk Management: Standards of Care in Diabetes—2026. Diabetes Care 2026;49(Suppl 1)"
SRC_USPSTF = "US Preventive Services Task Force. Screening for Prediabetes and Type 2 Diabetes (2021), khuyến cáo B: người lớn 35–70 tuổi thừa cân/béo phì (JAMA 2021;326(8):736-743)"
SRC_CDC = "CDC. National Diabetes Prevention Program – lifestyle change program (giảm 5–7% cân nặng, ≥150 phút hoạt động thể chất/tuần)"
SRC_TEAM = "Nhóm tự xây dựng từ kết quả thực nghiệm của đề tài (báo cáo Giai đoạn 2, mục 8–9; Giai đoạn 3, mục 3.1 và 3.3)"
VERIFY_NOTE = "Nội dung đã đối chiếu qua nguồn thứ cấp; nhóm cần đối chiếu số khuyến cáo/trang trong bản gốc trước khi nộp."

# Bảng đối chiếu tài liệu gốc (document, section, rec_no, status). rec_no để trống = nhóm chưa đối chiếu được số khuyến cáo trong bản gốc.
STATUS_PENDING = "Chưa đối chiếu bản gốc (mới qua nguồn thứ cấp) – nhóm điền số khuyến cáo"
STATUS_SECONDARY = "Đã đối chiếu bản tóm tắt AAFP; chờ đối chiếu bản gốc JAMA 2021"
STATUS_OPERATIONAL = "Không áp dụng – quy tắc vận hành của nhóm, kiểm chứng bằng test case"
_ADA2 = ("ADA Standards of Care 2026", "Mục 2 – Diagnosis and Classification of Diabetes (tiêu chí tầm soát)")
_ADA3 = ("ADA Standards of Care 2026", "Mục 3 – Prevention or Delay of Diabetes and Associated Comorbidities")
_ADA5 = ("ADA Standards of Care 2026", "Mục 5 – Facilitating Positive Health Behaviors and Well-being")
_ADA10 = ("ADA Standards of Care 2026", "Mục 10 – Cardiovascular Disease and Risk Management")
_CDC = ("CDC National Diabetes Prevention Program", "Chương trình thay đổi lối sống (giảm 5–7% cân nặng, ≥150 phút/tuần)")
_OP = ("Nhóm xây dựng", "Báo cáo Giai đoạn 2 mục 8–9; Giai đoạn 3 mục 3.1, 3.3")
RULE_META = {
    "R01": (*_OP, "", STATUS_OPERATIONAL), "R02": (*_OP, "", STATUS_OPERATIONAL),
    "R03": (*_OP, "", STATUS_OPERATIONAL),
    "R04": (*_ADA2, "", STATUS_PENDING),
    "R05": ("USPSTF 2021 (JAMA 2021;326(8):736-743)", "Recommendation Statement – khuyến cáo B: 35–70 tuổi, thừa cân/béo phì", "B", STATUS_SECONDARY),
    "R06": (*_ADA2, "", STATUS_PENDING),
    "R07": (*_CDC, "", STATUS_PENDING), "R08": (*_CDC, "", STATUS_PENDING),
    "R09": (*_ADA5, "", STATUS_PENDING), "R10": (*_ADA2, "", STATUS_PENDING),
    "R11": (*_ADA10, "", STATUS_PENDING), "R12": (*_ADA2, "", STATUS_PENDING),
    "R13": (*_ADA5, "", STATUS_PENDING),
    "R14": (*_OP, "", STATUS_OPERATIONAL), "R15": (*_OP, "", STATUS_OPERATIONAL), "R16": (*_OP, "", STATUS_OPERATIONAL),
}
R05_BOUNDARY = ("USPSTF áp dụng cho 35–70 tuổi, còn BRFSS chỉ có nhóm tuổi 5 năm: mã 4 = 35–39 … mã 10 = 65–69, mã 11 = 70–74. "
                "Quy tắc dùng mã 4–10 (35–69 tuổi) nên bỏ sót người đúng 70 tuổi (không tách được khỏi nhóm 70–74) – chọn cách này để không khuyến cáo vượt phạm vi 35–70. "
                "Hệ quả: R05 hơi hẹp hơn khuyến cáo gốc; người 70–74 vẫn nhận R04 (ADA, từ 35 tuổi).")
def apply_meta(rules):
    for r in rules["rules"]:
        d, s, n, st = RULE_META[r["id"]]
        r["doc"], r["section"], r["rec_no"], r["status"] = d, s, n, st
        if r["id"] == "R05":
            r["boundary_note"] = R05_BOUNDARY
    return rules


def _c(field, op, value):
    return {"field": field, "op": op, "value": value}


def default_rules() -> dict:
    R = []
    def add(id_, name, typ, prio, when, action, expl, source, verified=""):
        R.append({"id": id_, "name": name, "type": typ, "priority": prio, "when": when, "action": action, "explanation": expl,
                  "source": source, "verification": verified})
    add("R01", "Mức nguy cơ thấp – luôn kèm lưu ý giới hạn", "operational", 1, _c("tier", "==", "low"),
        "Duy trì lối sống lành mạnh và tầm soát định kỳ theo độ tuổi.",
        "Mức thấp CHỈ có nghĩa là mô hình thấy ít điểm giống nhóm đã được chẩn đoán trong dữ liệu khảo sát; KHÔNG có nghĩa chắc chắn không mắc bệnh "
        "(nhãn dữ liệu là thông tin tự khai về tiền sử chẩn đoán; mô hình vẫn bỏ sót một phần ca bệnh).", SRC_TEAM)
    add("R02", "Mức nguy cơ trung bình", "operational", 2, _c("tier", "==", "medium"),
        "Nên trao đổi với nhân viên y tế về xét nghiệm đường huyết (HbA1c hoặc đường huyết lúc đói) khi có điều kiện; tăng cường lối sống.",
        "Xác suất vượt ngưỡng Recall ≥ 80% (chọn trên validation) nhưng chưa tới ngưỡng F1 lớn nhất.", SRC_TEAM)
    add("R03", "Mức nguy cơ cao", "operational", 3, _c("tier", "==", "high"),
        "Ưu tiên để nhân viên y tế đánh giá và làm xét nghiệm xác nhận (HbA1c, đường huyết lúc đói hoặc nghiệm pháp dung nạp glucose 75 g).",
        "Xác suất vượt ngưỡng F1 lớn nhất; trong thực nghiệm nhóm này chiếm khoảng 1/5 hồ sơ nhưng chứa phần lớn ca có nhãn dương.",
        SRC_TEAM + "; danh sách xét nghiệm theo " + SRC_ADA2, VERIFY_NOTE)
    add("R04", "Tầm soát định kỳ từ 35 tuổi", "guideline", 2, _c("Age", ">=", 4),
        "Người từ 35 tuổi trở lên nên được tầm soát tiền đái tháo đường/đái tháo đường; nếu bình thường, lặp lại ít nhất mỗi 3 năm.",
        "ADA khuyến cáo bắt đầu tầm soát ở mọi người lớn từ 35 tuổi (mã tuổi ≥ 4 trong BRFSS); không phụ thuộc kết quả mô hình.", SRC_ADA2, VERIFY_NOTE)
    add("R05", "Tầm soát người 35–70 tuổi thừa cân", "guideline", 2,
        {"all": [_c("BMI", ">=", 25), _c("Age", ">=", 4), _c("Age", "<=", 10)]},
        "Người 35–70 tuổi có thừa cân/béo phì nên được tầm soát; nếu phát hiện tiền đái tháo đường, nên được giới thiệu can thiệp phòng ngừa.",
        "USPSTF 2021 (khuyến cáo B) áp dụng cho 35–70 tuổi có thừa cân/béo phì (BMI ≥ 25); mã tuổi 4–10 tương ứng 35–69 tuổi (người đúng 70 tuổi nằm trong mã 11 nên không được tính – xấp xỉ thận trọng).",
        SRC_USPSTF, "Đã đối chiếu với bản tóm tắt của AAFP về khuyến cáo USPSTF 2021.")
    add("R06", "Thừa cân kèm yếu tố nguy cơ (mọi độ tuổi)", "guideline", 2,
        {"all": [_c("BMI", ">=", 25), {"any": [_c("HighBP", "==", 1), _c("HeartDiseaseorAttack", "==", 1), _c("PhysActivity", "==", 0)]}]},
        "Nên được xét nghiệm đường huyết ở bất kỳ độ tuổi nào khi thừa cân/béo phì kèm ít nhất một yếu tố nguy cơ.",
        "ADA: tầm soát ở mọi tuổi khi thừa cân/béo phì kèm ≥ 1 yếu tố nguy cơ (ở đây: huyết áp cao, bệnh tim mạch, ít vận động). "
        "Dữ liệu không có chủng tộc nên không áp dụng ngưỡng BMI ≥ 23 cho người Mỹ gốc Á.", SRC_ADA2, VERIFY_NOTE)
    add("R07", "Quản lý cân nặng khi BMI ≥ 25", "guideline", 2, _c("BMI", ">=", 25),
        "Nếu được xác nhận tiền đái tháo đường/nguy cơ cao, nên tham gia chương trình thay đổi lối sống có cấu trúc (mục tiêu giảm 5–7% cân nặng).",
        "Mục tiêu giảm cân của chương trình phòng ngừa đái tháo đường quốc gia (DPP); ADA nêu mục tiêu tối thiểu khoảng 7%.", SRC_CDC + "; " + SRC_ADA3, VERIFY_NOTE)
    add("R08", "Ít vận động", "guideline", 2, _c("PhysActivity", "==", 0),
        "Nên tăng dần hoạt động thể chất cường độ vừa, mục tiêu khoảng 150 phút/tuần, sau khi hỏi ý kiến nhân viên y tế nếu có bệnh nền.",
        "Mục tiêu hoạt động thể chất của chương trình DPP và khuyến cáo ADA cho người nguy cơ cao.", SRC_CDC + "; " + SRC_ADA3, VERIFY_NOTE)
    add("R09", "Ăn ít rau/trái cây", "guideline", 1, {"any": [_c("Fruits", "==", 0), _c("Veggies", "==", 0)]},
        "Nên tăng rau và trái cây trong chế độ ăn hằng ngày.", "ADA khuyến khích mô hình ăn uống lành mạnh nhiều rau, trái cây cho người nguy cơ cao.", SRC_ADA5, VERIFY_NOTE)
    add("R10", "Huyết áp cao", "guideline", 2, _c("HighBP", "==", 1),
        "Nên theo dõi và kiểm soát huyết áp cùng nhân viên y tế.",
        "Tăng huyết áp nằm trong danh sách yếu tố nguy cơ của tiêu chí tầm soát ADA.", SRC_ADA2, VERIFY_NOTE)
    add("R11", "Cholesterol cao", "guideline", 1, _c("HighChol", "==", 1),
        "Nên được nhân viên y tế đánh giá nguy cơ tim mạch – chuyển hóa tổng thể.",
        "Rối loạn lipid thường đi cùng nguy cơ tim mạch – chuyển hóa; không dùng để kết luận về tiểu đường.", SRC_ADA10, VERIFY_NOTE)
    add("R12", "Có bệnh tim mạch", "guideline", 2, _c("HeartDiseaseorAttack", "==", 1),
        "Nên được xét nghiệm đường huyết bất kể mức nguy cơ của mô hình.",
        "Tiền sử bệnh tim mạch là một yếu tố nguy cơ trong tiêu chí tầm soát của ADA; kết quả mô hình thấp không thay thế khuyến cáo này.", SRC_ADA2, VERIFY_NOTE)
    add("R13", "Tiền sử hút thuốc (lưu ý cách hiểu biến)", "guideline", 1, _c("Smoker", "==", 1),
        "Biến này chỉ cho biết từng hút từ 100 điếu, không cho biết hiện còn hút. Nếu hiện vẫn hút thuốc, nên được hỗ trợ cai thuốc.",
        "Cai thuốc lá là một phần tư vấn hành vi sức khỏe; hệ thống không giả định người dùng đang hút thuốc.", SRC_ADA5, VERIFY_NOTE)
    add("R14", "Sức khỏe tự đánh giá yếu hoặc khó đi bộ", "operational", 2, {"any": [_c("GenHlth", ">=", 4), _c("DiffWalk", "==", 1)]},
        "Nên gặp nhân viên y tế để được đánh giá sức khỏe tổng quát.",
        "Trong dữ liệu, hai yếu tố này gắn với tỷ lệ nhãn dương cao hơn (quan hệ thống kê, không phải nhân quả).", SRC_TEAM)
    add("R15", "Rào cản tiếp cận y tế", "operational", 1, {"any": [_c("AnyHealthcare", "==", 0), _c("NoDocbcCost", "==", 1)]},
        "Nên được cung cấp thông tin về nơi tầm soát chi phí thấp tại địa phương.",
        "Người gặp rào cản chi phí/bảo hiểm ít có cơ hội kiểm tra định kỳ.", SRC_TEAM)
    add("R16", "Nhóm mô hình dễ bỏ sót", "operational", 2,
        {"all": [_c("tier", "==", "low"), _c("BMI", "<", 25), _c("HighBP", "==", 0)]},
        "Kết quả thấp không loại trừ bệnh; nếu có triệu chứng (khát nhiều, tiểu nhiều, mệt, sụt cân) hoặc yếu tố nguy cơ khác, nên đi khám.",
        "Báo cáo Giai đoạn 2 (mục 9): nhóm BMI < 25 và không có huyết áp cao bị mô hình bỏ sót nhiều hơn mức chung.", SRC_TEAM)
    return apply_meta({"version": "1.1", "description": "Bộ quy tắc hỗ trợ quyết định, độc lập với mô hình học máy. Quy tắc 'guideline' có nguồn chuyên môn; 'operational' do nhóm xây dựng từ thực nghiệm. "
            "Các biến HvyAlcoholConsump không được dùng làm căn cứ cho bất kỳ lời khuyên nào.", "rules": R})


def save_rules(rules: dict | None = None) -> dict:
    rules = rules or default_rules()
    RULES_PATH.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> Đã lưu bộ quy tắc ({len(rules['rules'])} quy tắc): {RULES_PATH}")
    return rules


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_OPS = {"==": np.equal, "!=": np.not_equal, ">=": np.greater_equal, "<=": np.less_equal, ">": np.greater, "<": np.less}


def eval_condition(cond: dict, data: dict) -> np.ndarray:
    """Đánh giá điều kiện (all/any/điều kiện đơn) trên dict {trường: mảng}; trả về mảng bool (vectorized)."""
    if "all" in cond:
        out = eval_condition(cond["all"][0], data)
        for c in cond["all"][1:]:
            out = out & eval_condition(c, data)
        return out
    if "any" in cond:
        out = eval_condition(cond["any"][0], data)
        for c in cond["any"][1:]:
            out = out | eval_condition(c, data)
        return out
    return _OPS[cond["op"]](np.asarray(data[cond["field"]]), cond["value"])


def _fields(cond: dict) -> list:
    if "all" in cond or "any" in cond:
        return [f for c in cond.get("all", cond.get("any")) for f in _fields(c)]
    return [cond["field"]]


def rule_matrix(rules: dict, X: pd.DataFrame, tier) -> pd.DataFrame:
    data = {c: X[c].to_numpy() for c in X.columns}
    data["tier"] = np.asarray(tier)
    return pd.DataFrame({r["id"]: eval_condition(r["when"], data) for r in rules["rules"]}, index=X.index)


def decide(features: dict, p: float, thresholds: dict, rules: dict | None = None) -> dict:
    """Quyết định cho một hồ sơ: mức nguy cơ (từ xác suất) + các quy tắc được kích hoạt + mức ưu tiên hỗ trợ."""
    rules = rules or load_rules()
    tier = str(assign_tier([p], thresholds["t_low"], thresholds["t_high"])[0])
    X = pd.DataFrame([{k: features[k] for k in FEATURES}])
    mat = rule_matrix(rules, X, [tier]).iloc[0]
    fired = [r for r in rules["rules"] if bool(mat[r["id"]])]
    prio = max([TIER_PRIORITY[tier]] + [r["priority"] for r in fired])
    return {"tier": tier, "tier_name": TIER_NAMES[tier], "priority": prio, "priority_name": PRIORITY_NAMES[prio], "p": float(p),
            "rules": fired, "rule_ids": [r["id"] for r in fired],
            "disclaimer": "Kết quả dựa trên dữ liệu khảo sát tự khai (BRFSS 2015), chỉ mang tính tham khảo, không phải chẩn đoán và không thay thế đánh giá của nhân viên y tế."
            + (" Mức thấp không có nghĩa chắc chắn không mắc bệnh." if tier == "low" else "")}


def validate_rules(rules: dict, thresholds: dict | None = None) -> pd.DataFrame:
    """Kiểm tra tính nhất quán của bộ quy tắc: trả về bảng (kiểm tra, kết quả, chi tiết)."""
    R = rules["rules"]
    res = []
    def add(name, ok, detail=""):
        res.append({"Kiểm tra": name, "Kết quả": "ĐẠT" if ok else "LỖI", "Chi tiết": detail})
    ids = [r["id"] for r in R]
    add("Mã quy tắc không trùng", len(ids) == len(set(ids)), "")
    need = {"id", "name", "type", "priority", "when", "action", "explanation", "source"}
    add("Đủ trường bắt buộc", all(need <= set(r) for r in R), "")
    add("Quy tắc 'guideline' đều có nguồn tham khảo", all(r["source"] for r in R if r["type"] == "guideline"), "")
    allowed = set(FEATURES) | {"tier"}
    bad = [r["id"] for r in R if not set(_fields(r["when"])) <= allowed]
    add("Điều kiện chỉ dùng biến hợp lệ", not bad, ", ".join(bad))
    add("Quy tắc 'guideline' có tài liệu, mục và trạng thái xác minh", all(r.get("doc") and r.get("section") and r.get("status") for r in R if r["type"] == "guideline"), "")
    add("Có 10 quy tắc guideline và 6 quy tắc operational", sum(r["type"] == "guideline" for r in R) == 10 and sum(r["type"] == "operational" for r in R) == 6, "")
    add("Không quy tắc nào dùng HvyAlcoholConsump làm căn cứ", all("HvyAlcoholConsump" not in _fields(r["when"]) for r in R), "")
    dup = [(a["id"], b["id"]) for i, a in enumerate(R) for b in R[i + 1:] if a["when"] == b["when"]]
    add("Không có hai quy tắc trùng điều kiện", not dup, str(dup))
    grid = np.linspace(0, 1, 2001)
    th = thresholds or {"t_low": 0.134, "t_high": 0.238}
    t = assign_tier(grid, th["t_low"], th["t_high"])
    dummy = pd.DataFrame({f: np.zeros(len(grid), dtype=int) for f in FEATURES})
    m = rule_matrix(rules, dummy, t)
    n_tier = m[[r["id"] for r in R if r["id"] in ("R01", "R02", "R03")]].sum(axis=1)
    add("Đúng một quy tắc mức nguy cơ được kích hoạt với mọi xác suất", bool((n_tier == 1).all()), "")
    low_ok = bool(m.loc[t == "low", "R01"].all())
    add("Mức thấp luôn kèm lưu ý giới hạn (R01)", low_ok, "")
    smoke = next((r for r in R if r["id"] == "R13"), None)
    add("Quy tắc Smoker không giả định đang hút thuốc", smoke is not None and "không cho biết hiện còn hút" in smoke["action"], "")
    return pd.DataFrame(res)


def _neutral() -> dict:
    return {"HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
            "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0,
            "Sex": 0, "Age": 3, "Education": 5, "Income": 6}


def test_cases(thresholds: dict | None = None) -> pd.DataFrame:
    """23 tình huống kiểm thử; kết quả kỳ vọng được viết tay từ định nghĩa quy tắc (độc lập với engine)."""
    N = _neutral()
    th = thresholds or {"t_low": 0.134, "t_high": 0.238}   # mặc định chỉ để tạo file mẫu; kiểm thử thật truyền ngưỡng lấy từ Phase3_system.joblib
    tl, tt = float(th["t_low"]), float(th["t_high"])
    C = [
        ("T01", "Người khỏe mạnh, xác suất rất thấp", {}, 0.03, "low", {"R01", "R16"}),
        ("T02", "Người khỏe mạnh, xác suất trong vùng trung bình", {}, 0.20, "medium", {"R02"}),
        ("T03", "Người khỏe mạnh, xác suất cao", {}, 0.40, "high", {"R03"}),
        ("T04", "35–39 tuổi, mức thấp (kiểm tra tầm soát theo tuổi)", {"Age": 4}, 0.05, "low", {"R01", "R04", "R16"}),
        ("T05", "Nhiều yếu tố: 45–49 tuổi, BMI 27, huyết áp cao, cholesterol cao", {"Age": 6, "BMI": 27, "HighBP": 1, "HighChol": 1}, 0.30, "high", {"R03", "R04", "R05", "R06", "R07", "R10", "R11"}),
        ("T06", "Nhiều yếu tố: 60–64 tuổi, BMI 33, ít vận động, ăn ít rau quả, từng hút thuốc", {"Age": 9, "BMI": 33, "PhysActivity": 0, "Smoker": 1, "HighBP": 1, "HighChol": 1, "Fruits": 0}, 0.55, "high", {"R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R13"}),
        ("T07", "75–79 tuổi, BMI 24, có bệnh tim, mức thấp", {"Age": 12, "BMI": 24, "HeartDiseaseorAttack": 1}, 0.10, "low", {"R01", "R04", "R12", "R16"}),
        ("T08", "Hồ sơ dễ bị bỏ sót: trẻ, BMI 22, không huyết áp cao, sức khỏe khá", {"Age": 2, "GenHlth": 3}, 0.055, "low", {"R01", "R16"}),
        ("T09", "Sát dưới ngưỡng thấp/trung bình (p = t_low − 0,0001)", {"Age": 5, "BMI": 26}, tl - 1e-4, "low", {"R01", "R04", "R05", "R07"}),
        ("T10", "Đúng ngưỡng thấp/trung bình (p = t_low)", {"Age": 5, "BMI": 26}, tl, "medium", {"R02", "R04", "R05", "R07"}),
        ("T11", "Sát dưới ngưỡng trung bình/cao (p = t_high − 0,0001)", {"Age": 7, "BMI": 31, "HighBP": 1}, tt - 1e-4, "medium", {"R02", "R04", "R05", "R06", "R07", "R10"}),
        ("T12", "Đúng ngưỡng trung bình/cao (p = t_high)", {"Age": 7, "BMI": 31, "HighBP": 1}, tt, "high", {"R03", "R04", "R05", "R06", "R07", "R10"}),
        ("T13", "Rào cản tiếp cận y tế, mức thấp", {"AnyHealthcare": 0, "NoDocbcCost": 1, "BMI": 24}, 0.08, "low", {"R01", "R15", "R16"}),
        ("T14", "70–74 tuổi, sức khỏe kém, khó đi bộ, huyết áp cao", {"Age": 11, "GenHlth": 5, "DiffWalk": 1, "BMI": 29, "HighBP": 1}, 0.45, "high", {"R03", "R04", "R06", "R07", "R10", "R14"}),
        ("T15", "Chỉ có tiền sử hút thuốc, mức thấp", {"Smoker": 1, "BMI": 23}, 0.04, "low", {"R01", "R13", "R16"}),
        ("T16", "Uống rượu nhiều – không được sinh khuyến nghị nào theo rượu bia", {"HvyAlcoholConsump": 1}, 0.02, "low", {"R01", "R16"}),
        ("T17", "BMI 40, ít vận động, ăn ít rau quả, cholesterol cao", {"Age": 8, "BMI": 40, "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "HighChol": 1}, 0.35, "high", {"R03", "R04", "R05", "R06", "R07", "R08", "R09", "R11"}),
        ("T18", "Biên BMI = 25 đúng, 35–39 tuổi, không yếu tố nguy cơ khác", {"Age": 4, "BMI": 25}, 0.15, "medium", {"R02", "R04", "R05", "R07"}),
        ("T19", "80 tuổi trở lên, BMI 21, mức thấp", {"Age": 13, "BMI": 21}, 0.06, "low", {"R01", "R04", "R16"}),
        ("T20", "Ranh giới R05: 65–69 tuổi (mã 10), BMI 26 → R05 áp dụng", {"Age": 10, "BMI": 26}, 0.20, "medium", {"R02", "R04", "R05", "R07"}),
        ("T21", "BMI 24,9 (sát dưới 25), 35–39 tuổi → không R05/R06/R07", {"Age": 4, "BMI": 24.9}, 0.15, "medium", {"R02", "R04"}),
        ("T22", "30–34 tuổi, BMI 30, không yếu tố nguy cơ → chỉ R07 (chưa đủ tuổi R04/R05)", {"Age": 3, "BMI": 30}, 0.05, "low", {"R01", "R07"}),
        ("T23", "Ranh giới R05: 70–74 tuổi (mã 11), BMI 26 → R05 không áp dụng, vẫn có R04", {"Age": 11, "BMI": 26}, 0.20, "medium", {"R02", "R04", "R07"}),
    ]
    rows = []
    for cid, desc, over, p, tier, exp in C:
        rows.append({"id": cid, "mô tả": desc, "p": p, **{**N, **over}, "mức kỳ vọng": tier, "quy tắc kỳ vọng": ",".join(sorted(exp))})
    return pd.DataFrame(rows)


def run_rule_tests(rules: dict, thresholds: dict | None = None):
    """Chạy các tình huống kiểm thử cho engine. Trả về (bảng kết quả, chỉ số tổng hợp)."""
    thresholds = thresholds or {"t_low": 0.134, "t_high": 0.238}   # mặc định chỉ khi chưa có hệ thống; hãy truyền system["thresholds"]
    cases = test_cases(thresholds)
    cases.to_csv(TESTCASE_PATH, index=False, encoding="utf-8-sig")
    rows, times = [], []
    for _, c in cases.iterrows():
        t0 = time.perf_counter()
        d = decide({f: c[f] for f in FEATURES}, c["p"], thresholds, rules)
        times.append(1000 * (time.perf_counter() - t0))
        exp = set(c["quy tắc kỳ vọng"].split(","))
        got = set(d["rule_ids"])
        rows.append({"id": c["id"], "mô tả": c["mô tả"], "p": c["p"], "Mức kỳ vọng": TIER_NAMES[c["mức kỳ vọng"]], "Mức engine": d["tier_name"],
                     "Quy tắc kỳ vọng": ",".join(sorted(exp)), "Quy tắc kích hoạt": ",".join(sorted(got)),
                     "Thiếu": ",".join(sorted(exp - got)), "Thừa": ",".join(sorted(got - exp)),
                     "Đạt": d["tier"] == c["mức kỳ vọng"] and exp == got,
                     "Nhất quán (ưu tiên ≥ mức nguy cơ; mức thấp có R01)": d["priority"] >= TIER_PRIORITY[d["tier"]] and (d["tier"] != "low" or "R01" in got)})
    res = pd.DataFrame(rows)
    conflicts = validate_rules(rules, thresholds)
    summ = {"Số tình huống": len(res), "Tỷ lệ tình huống đạt (mức + quy tắc kích hoạt đúng)": float(res["Đạt"].mean()),
            "Tỷ lệ nhất quán mô hình–mức cảnh báo–quy tắc": float(res.iloc[:, -1].mean()),
            "Số kiểm tra tính nhất quán bộ quy tắc đạt": f"{(conflicts['Kết quả'] == 'ĐẠT').sum()}/{len(conflicts)}",
            "Thời gian quyết định trung bình (ms/hồ sơ, chưa gồm mô hình)": float(np.mean(times))}
    _save_table(res, "3_5_rule_tests.csv")
    _save_table(conflicts, "3_5_rule_consistency.csv")
    return res, conflicts, summ


# ----------------------------------------------------------------------------------------------
# 3.6  What-if có ràng buộc
# ----------------------------------------------------------------------------------------------
MUTABLE = {"BMI": "Giảm BMI (không tăng; trong khoảng 15–60)", "PhysActivity": "Bắt đầu vận động (0 → 1)",
           "Fruits": "Ăn trái cây hằng ngày (0 → 1)", "Veggies": "Ăn rau hằng ngày (0 → 1)"}
LOCKED_REASON = {
    "Age": "tuổi không thể can thiệp", "Sex": "giới tính không thể can thiệp", "Stroke": "tiền sử đột quỵ không thể thay đổi",
    "HeartDiseaseorAttack": "tiền sử bệnh tim không thể thay đổi", "Smoker": "biến này là 'đã từng hút ≥100 điếu', không phải tình trạng hiện tại – không được chuyển 1 → 0 để mô phỏng cai thuốc",
    "HvyAlcoholConsump": "không mô phỏng thay đổi theo hướng này vì quan hệ nghịch trong dữ liệu có thể do người bệnh bỏ rượu (không dùng làm lời khuyên)",
    "HighBP": "là tình trạng bệnh/chẩn đoán, không phải hành vi can thiệp", "HighChol": "là tình trạng bệnh/chẩn đoán, không phải hành vi can thiệp",
    "GenHlth": "là tự đánh giá, không phải biến can thiệp trực tiếp", "MentHlth": "không phải biến can thiệp trong phạm vi đề tài", "PhysHlth": "không phải biến can thiệp trong phạm vi đề tài",
    "DiffWalk": "là tình trạng sức khỏe", "CholCheck": "không phải biến can thiệp", "AnyHealthcare": "yếu tố xã hội, không mô phỏng", "NoDocbcCost": "yếu tố xã hội, không mô phỏng",
    "Education": "không thể can thiệp", "Income": "không thể can thiệp",
}
WHATIF_DISCLAIMER = ("What-if chỉ mô phỏng phản ứng của mô hình khi đầu vào thay đổi; KHÔNG chứng minh rằng một hành động cụ thể sẽ làm giảm nguy cơ mắc bệnh "
                     "đúng theo mức thay đổi của xác suất. Dữ liệu là khảo sát cắt ngang, quan hệ là thống kê, không phải nhân quả.")


def validate_changes(features: dict, changes: dict) -> dict:
    """Kiểm tra thay đổi đầu vào theo quy định; ném ValueError nếu vi phạm. Trả về dict cảnh báo (không chặn)."""
    warns = []
    for k, v in changes.items():
        if k not in FEATURES:
            raise ValueError(f"'{k}' không phải biến của mô hình")
        if k not in MUTABLE:
            raise ValueError(f"Không được thay đổi '{k}': {LOCKED_REASON.get(k, 'không phải biến can thiệp')}")
        if k == "BMI":
            if not (15.0 <= v <= 60.0):
                raise ValueError("BMI sau can thiệp phải nằm trong khoảng 15–60")
            if v > features["BMI"]:
                raise ValueError("What-if chỉ cho phép giảm BMI")
            if v < 18.5:
                warns.append("BMI sau can thiệp dưới 18,5 (thiếu cân) – không phải mục tiêu phù hợp")
            if features["BMI"] > 0 and (features["BMI"] - v) / features["BMI"] > 0.10:
                warns.append("Mức giảm BMI > 10% vượt mức mục tiêu thông thường (5–7%, tối đa khoảng 10%) của chương trình DPP")
        elif v not in (0, 1) or v < features[k]:
            raise ValueError(f"'{k}' chỉ được chuyển từ 0 sang 1")
    return {"warnings": warns}


def support_count(X_ref: pd.DataFrame, features: dict, bmi_tol: float = 2.0) -> int:
    """Số hồ sơ huấn luyện giống hồ sơ (cùng tuổi, giới, huyết áp cao; BMI trong ±2): đo mức hợp lý của tổ hợp đầu vào."""
    m = (X_ref["Age"] == features["Age"]) & (X_ref["Sex"] == features["Sex"]) & (X_ref["HighBP"] == features["HighBP"]) & ((X_ref["BMI"] - features["BMI"]).abs() <= bmi_tol)
    return int(m.sum())


# ----------------------------------------------------------------------------------------------
# Hệ thống đóng gói (dùng cho giao diện) và các hàm API
# ----------------------------------------------------------------------------------------------
def build_system(fin: dict, scheme_key: str, calibration_note: str, summary_cfg: dict) -> dict:
    t_low, t_high = tier_schemes(fin)[scheme_key]
    sysd = {"model": fin["model"], "calibrator": fin["calibrator"], "calibration": fin["calibration"], "calibration_note": calibration_note,
            "thresholds": {"t_low": float(t_low), "t_high": float(t_high), "scheme": scheme_key, "ops": {k: float(v) for k, v in fin["ops"].items()}},
            "features": FEATURES, "X_ref": fin["X_train"], "seed": fin["seed"], "model_name": "LightGBM", "strategy": summary_cfg["strategy"],
            "target_meaning": "Xác suất thuộc nhóm có tiền sử được chẩn đoán tiền tiểu đường/tiểu đường theo dữ liệu khảo sát BRFSS 2015 (tự khai) – không phải xác suất mắc bệnh trong tương lai."}
    joblib.dump(sysd, SYSTEM_PATH, compress=3)
    print(f"-> Đã lưu hệ thống: {SYSTEM_PATH}")
    return sysd


def load_system(path: Path = SYSTEM_PATH) -> dict:
    return joblib.load(path)


def _prob(system: dict, X: pd.DataFrame) -> np.ndarray:
    return system["calibrator"].transform(system["model"].predict_proba(X[FEATURES])[:, 1])


def assess(system: dict, features: dict, rules: dict | None = None, top: int = 6) -> dict:
    """Một lần đánh giá: xác suất → mức nguy cơ → quy tắc → giải thích SHAP."""
    t0 = time.perf_counter()
    X = pd.DataFrame([{k: features[k] for k in FEATURES}])
    p = float(_prob(system, X)[0])
    d = decide(features, p, system["thresholds"], rules)
    sv, base = contribs(system["model"], X)
    order = np.argsort(-np.abs(sv[0]))[:top]
    d["contributions"] = pd.DataFrame({"Đặc trưng": [FEATURES[j] for j in order], "Tên": [FEATURE_VN[FEATURES[j]] for j in order],
                                       "Giá trị": [features[FEATURES[j]] for j in order], "SHAP (log-odds thô)": sv[0][order]})
    d["shap_base_logodds"], d["shap_total_logodds"] = float(base[0]), float(base[0] + sv[0].sum())
    d["model_note"] = system["target_meaning"]
    d["elapsed_ms"] = 1000 * (time.perf_counter() - t0)
    return d


def what_if(system: dict, features: dict, changes: dict, rules: dict | None = None) -> dict:
    chk = validate_changes(features, changes)
    after = {**features, **changes}
    a0, a1 = assess(system, features, rules), assess(system, after, rules)
    sv0, _ = contribs(system["model"], pd.DataFrame([features])[FEATURES])
    sv1, _ = contribs(system["model"], pd.DataFrame([after])[FEATURES])
    diff = (sv1 - sv0)[0]
    order = np.argsort(-np.abs(diff))[:5]
    why = pd.DataFrame({"Đặc trưng": [FEATURES[j] for j in order], "Trước": [features[FEATURES[j]] for j in order],
                        "Sau": [after[FEATURES[j]] for j in order], "Thay đổi SHAP (log-odds thô)": diff[order]})
    n_sup = support_count(pd.DataFrame(system["X_ref"]), after)
    flags = list(chk["warnings"])
    if n_sup < 30:
        flags.append(f"Tổ hợp đầu vào sau thay đổi hiếm trong dữ liệu huấn luyện (chỉ {n_sup} hồ sơ tương tự) – kết quả mô phỏng kém tin cậy")
    return {"before": a0, "after": a1, "delta_p": a1["p"] - a0["p"], "why": why, "support": n_sup, "flags": flags, "disclaimer": WHATIF_DISCLAIMER}


def whatif_scenarios(fin: dict, system: dict, t_low: float, t_high: float):
    """Mô phỏng nhóm: hồ sơ test ở mức cao, BMI ≥ 25 và ít vận động. Kèm đường cong giảm BMI để thấy hiệu ứng bước ngưỡng của cây."""
    X, p0 = fin["X_test"].copy(), fin["p_test"]
    m = (assign_tier(p0, t_low, t_high) == "high") & (X["BMI"] >= 25).to_numpy() & (X["PhysActivity"] == 0).to_numpy()
    G = X[m].reset_index(drop=True)
    def after(df, pct=0.0, life=False):
        d = df.copy()
        if pct:
            d["BMI"] = np.clip(d["BMI"] * (1 - pct), 15.0, 60.0)
        if life:
            d["PhysActivity"], d["Fruits"], d["Veggies"] = 1, 1, 1
        return d
    scen = {"PA0: Hiện tại": (0, False), "PA1: Lối sống (vận động + rau + trái cây)": (0, True), "PA2: Giảm 5% BMI": (0.05, False),
            "PA3: Giảm 7% BMI (mục tiêu DPP)": (0.07, False), "PA4: Giảm 10% BMI": (0.10, False), "PA5: Giảm 7% BMI + lối sống": (0.07, True)}
    rows = []
    base_p = _prob(system, G)
    for name, (pct, life) in scen.items():
        pp = _prob(system, after(G, pct, life))
        t = assign_tier(pp, t_low, t_high)
        rows.append({"Kịch bản": name, "Số hồ sơ": len(G), "Xác suất TB": pp.mean(), "Δp tuyệt đối TB": (pp - base_p).mean(),
                     "Δp tương đối TB (%)": 100 * ((pp - base_p) / base_p).mean(), "% còn mức Cao": 100 * (t == "high").mean(),
                     "% xuống Trung bình": 100 * (t == "medium").mean(), "% xuống Thấp": 100 * (t == "low").mean()})
    tab = pd.DataFrame(rows)
    _save_table(tab, "3_6_whatif_scenarios.csv")
    pcts = np.linspace(0, 0.15, 16)
    curve = pd.DataFrame({"Giảm BMI (%)": 100 * pcts, "Xác suất TB": [float(_prob(system, after(G, c)).mean()) for c in pcts]})
    _save_table(curve, "3_6_whatif_bmi_curve.csv")
    return tab, curve, G


def plot_whatif(tab: pd.DataFrame, curve: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].barh(tab["Kịch bản"][::-1], tab["Xác suất TB"][::-1], color="#3b6ea5")
    for i, v in enumerate(tab["Xác suất TB"][::-1]):
        axes[0].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    axes[0].set_xlabel("Xác suất dự đoán trung bình"); axes[0].set_title("Phản ứng của mô hình theo kịch bản (nhóm mức Cao, BMI ≥ 25, ít vận động)", fontsize=10)
    axes[1].plot(curve["Giảm BMI (%)"], curve["Xác suất TB"], "o-", color="#d62728")
    axes[1].set_xlabel("Mức giảm BMI (%)"); axes[1].set_ylabel("Xác suất dự đoán trung bình"); axes[1].grid(alpha=0.3)
    axes[1].set_title("Đường cong giảm BMI – quan sát các bước nhảy (hiệu ứng ngưỡng của cây)", fontsize=10)
    fig.text(0.5, -0.02, WHATIF_DISCLAIMER, ha="center", fontsize=7, wrap=True)
    _save_fig(fig, "3_6_whatif.png")
    return fig


# ----------------------------------------------------------------------------------------------
# 3.7  Thí nghiệm so sánh cấu hình A, B, C, D
# ----------------------------------------------------------------------------------------------
def compare_configs(fins: list, rules: dict):
    rows = []
    for r in fins:
        y, p, n = r["y_test"], r["p_test"], len(r["y_test"])
        def add(cfg, name, m=None, k=None):
            if k is not None:
                tm = _topk_metrics(y, p, k)
                tp = tm["Precision@k"] * k
                fn, fp, alerts = len(y[y == 1]) - tp, k - tp, k
                rec, prec = tm["Recall@k"], tm["Precision@k"]
            else:
                fn, fp, alerts, rec, prec = m["FN"], m["FP"], m["TP"] + m["FP"], m["Recall"], m["Precision"]
            row = {"Seed": r["seed"], "Cấu hình": cfg, "Phương án": name, "Recall": rec, "Precision": prec, "FN / 1.000": 1000 * fn / n,
                   "FP / 1.000": 1000 * fp / n, "Hồ sơ xử lý / 1.000": 1000 * alerts / n}
            for c in (5, 10):
                row[f"Chi phí / 1.000 (FN:FP={c}:1)"] = 1000 * (c * fn + fp) / n
            rows.append(row)
        add("A", "Ngưỡng mặc định 0,5", U.threshold_metrics(y, p, 0.5))
        add("B", "Ngưỡng chọn trên validation: Recall ≥ 80%", U.threshold_metrics(y, p, r["ops"]["Recall >= 80%"]))
        add("B", "Ngưỡng chọn trên validation: F1 lớn nhất", U.threshold_metrics(y, p, r["ops"]["F1 lớn nhất"]))
        add("C", "Theo chi phí giả định 5:1", U.threshold_metrics(y, p, r["ops"]["Chi phí FN:FP = 5:1"]))
        add("C", "Theo chi phí giả định 10:1", U.threshold_metrics(y, p, r["ops"]["Chi phí FN:FP = 10:1"]))
        add("C", "Theo năng lực: Top 200/1.000", k=round(0.2 * n))
        add("C", "Theo năng lực: Top 300/1.000", k=round(0.3 * n))
    d = pd.DataFrame(rows)
    cols = ["Recall", "Precision", "FN / 1.000", "FP / 1.000", "Hồ sơ xử lý / 1.000", "Chi phí / 1.000 (FN:FP=5:1)", "Chi phí / 1.000 (FN:FP=10:1)"]
    summ = U.summarize(d, ["Cấu hình", "Phương án"], cols, 3)
    order = list(dict.fromkeys(zip(d["Cấu hình"], d["Phương án"])))
    summ = summ.set_index(["Cấu hình", "Phương án"]).loc[order].reset_index()
    _save_table(summ, "3_7_configs_ABC.csv")
    return d, summ


def system_quality(fin: dict, system: dict, rules: dict, rule_test_summary: dict) -> pd.DataFrame:
    """Cấu hình D: đánh giá riêng chất lượng hỗ trợ quyết định (không dùng PR-AUC)."""
    X, y, p = fin["X_test"], fin["y_test"], fin["p_test"]
    thr = system["thresholds"]
    tier = assign_tier(p, thr["t_low"], thr["t_high"])
    mat = rule_matrix(rules, X, tier)
    prompt_ids = ["R04", "R05", "R06", "R12"]                       # các quy tắc nhắc tầm soát theo hướng dẫn chuyên môn
    prompt = mat[prompt_ids].any(axis=1).to_numpy()
    low = tier == "low"
    fn_low = low & (y == 1)
    sample = X.sample(min(300, len(X)), random_state=0)
    t0 = time.perf_counter()
    for i in range(len(sample)):
        assess(system, sample.iloc[i].to_dict(), rules)
    ms = 1000 * (time.perf_counter() - t0) / len(sample)
    rows = [
        ("Số tình huống kiểm thử quy tắc", rule_test_summary["Số tình huống"]),
        ("Tỷ lệ tình huống kích hoạt đúng quy tắc và mức", f"{100 * rule_test_summary['Tỷ lệ tình huống đạt (mức + quy tắc kích hoạt đúng)']:.1f}%"),
        ("Tỷ lệ nhất quán mô hình – mức cảnh báo – quy tắc", f"{100 * rule_test_summary['Tỷ lệ nhất quán mô hình–mức cảnh báo–quy tắc']:.1f}%"),
        ("Kiểm tra mâu thuẫn bộ quy tắc (đạt)", rule_test_summary["Số kiểm tra tính nhất quán bộ quy tắc đạt"]),
        ("Hồ sơ mức thấp luôn có lưu ý giới hạn (R01)", f"{100 * mat.loc[low, 'R01'].mean():.1f}%"),
        ("Ca bị bỏ sót (mức thấp, nhãn 1) được nhắc tầm soát theo hướng dẫn chuyên môn", f"{100 * prompt[fn_low].mean():.1f}% ({int(prompt[fn_low].sum())}/{int(fn_low.sum())})"),
        ("Hồ sơ mức thấp nhận nhắc tầm soát (chi phí vận hành của nhắc nhở)", f"{100 * prompt[low].mean():.1f}%"),
        ("Hồ sơ mức thấp thuộc nhóm dễ bỏ sót (R16)", f"{100 * mat.loc[low, 'R16'].mean():.1f}%"),
        ("Thời gian phản hồi trung bình cho một hồ sơ (gồm mô hình + SHAP + quy tắc)", f"{ms:.1f} ms"),
    ]
    out = pd.DataFrame(rows, columns=["Chỉ tiêu (cấu hình D, test seed 42)", "Kết quả"])
    _save_table(out, "3_7_system_quality_D.csv")
    return out


# ----------------------------------------------------------------------------------------------
# Xuất kết quả
# ----------------------------------------------------------------------------------------------
def export_frame(features: dict, result: dict, system: dict) -> pd.DataFrame:
    rows = [("Mô hình", f"{system['model_name']} (seed {system['seed']}), hiệu chuẩn: {U.CAL_LABEL[system['calibration']]}"),
            ("Ý nghĩa xác suất", system["target_meaning"]),
            ("Xác suất dự đoán", f"{result['p']:.3f}"), ("Mức nguy cơ", result["tier_name"]), ("Mức ưu tiên hỗ trợ", result["priority_name"]),
            ("Ngưỡng phân mức", f"thấp < {system['thresholds']['t_low']:.3f} ≤ trung bình < {system['thresholds']['t_high']:.3f} ≤ cao"),
            ("Sơ đồ phân mức", system["thresholds"]["scheme"])]
    rows += [(f"Đầu vào – {FEATURE_VN[k]}", VALUE_LABELS.get(k, {}).get(v, v) if k in VALUE_LABELS else (YESNO.get(v, v) if k in BINARY_FEATURES else v)) for k, v in features.items()]
    rows += [(f"Quy tắc {r['id']} ({'chuyên môn' if r['type'] == 'guideline' else 'vận hành'})", f"{r['action']} | Nguồn: {r['source']}") for r in result["rules"]]
    rows += [("Giới hạn sử dụng", result["disclaimer"])]
    return pd.DataFrame(rows, columns=["Mục", "Nội dung"])


# ----------------------------------------------------------------------------------------------
# 3.9  Decision Curve Analysis (tùy chọn, trên tập test, trung bình các seed)
# ----------------------------------------------------------------------------------------------
def decision_curve(fins: list, thresholds=None) -> pd.DataFrame:
    """Lợi ích ròng NB(pt) = TP/n − FP/n · pt/(1−pt) cho mô hình, 'cảnh báo tất cả' và 'không cảnh báo ai'.
    pt là ngưỡng xác suất mà tại đó người ra quyết định coi một ca bỏ sót nặng gấp (1−pt)/pt lần một cảnh báo sai.
    Đây là phân tích giả định trên dữ liệu khảo sát tự khai; KHÔNG phải bằng chứng hiệu quả lâm sàng."""
    thresholds = np.round(np.arange(0.05, 0.501, 0.01), 2) if thresholds is None else thresholds
    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        nb_m, nb_a = [], []
        for r in fins:
            y, p = np.asarray(r["y_test"]), np.asarray(r["p_test"]); n = len(y)
            pred = p >= pt
            tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
            nb_m.append(tp / n - fp / n * w)
            nb_a.append(y.mean() - (1 - y.mean()) * w)
        rows.append({"Ngưỡng pt": pt, "Mô hình (LightGBM)": float(np.mean(nb_m)), "Cảnh báo tất cả": float(np.mean(nb_a)), "Không cảnh báo ai": 0.0})
    out = pd.DataFrame(rows)
    _save_table(out, "3_9_decision_curve.csv")
    return out


def plot_decision_curve(dca: pd.DataFrame, t_low: float | None = None, t_high: float | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dca["Ngưỡng pt"], dca["Mô hình (LightGBM)"], color="#19A7B8", lw=2, label="LightGBM")
    ax.plot(dca["Ngưỡng pt"], dca["Cảnh báo tất cả"], color="#94A3B8", ls="--", label="Cảnh báo tất cả")
    ax.axhline(0, color="black", lw=1, label="Không cảnh báo ai")
    for t, nm in ((t_low, "ngưỡng Thấp/TB"), (t_high, "ngưỡng TB/Cao")):
        if t is not None:
            ax.axvline(t, color="grey", ls=":", lw=1); ax.text(t, ax.get_ylim()[1] * 0.92, nm, rotation=90, fontsize=7, va="top", ha="right")
    ax.set_ylim(-0.02, max(0.14, dca["Mô hình (LightGBM)"].max() * 1.1))
    ax.set_xlabel("Ngưỡng xác suất pt (mức đánh đổi giữa bỏ sót và cảnh báo sai)"); ax.set_ylabel("Lợi ích ròng"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save_fig(fig, "3_9_decision_curve.png")
    return fig
