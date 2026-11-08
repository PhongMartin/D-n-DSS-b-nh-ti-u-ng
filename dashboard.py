"""Dashboard và các màn hình kết quả cho app.py (INS305 – Nhóm 01).

Đọc trực tiếp các bảng CSV do notebook Giai đoạn 2–3 sinh ra (results/tables), nên không cần chạy lại mô hình.
Các ô trong bảng có dạng "trung bình ± sd [CI95 dưới; trên]" (xem U.summarize) – hàm num() lấy phần trung bình.
Mọi con số trên Dashboard là kết quả trên tập TEST (trung bình 5 seed) trừ khi ghi chú khác.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import dss_stage3 as S3
import dss_utils as U
import ui

TEAL, AQUA, INK, MUTED = "#0E5A66", "#19A7B8", "#0A2A33", "#52666B"
RED, AMBER, GREEN, BLUE = "#D64545", "#E9A93B", "#34A36B", "#0369A1"
TIER_COLORS = {"Thấp": GREEN, "Trung bình": AMBER, "Cao": RED}
OP_ORDER = ["Recall >= 90%", "Recall >= 80%", "F1 lớn nhất", "Ngưỡng mặc định 0,5"]
DEFAULT_OP = "Recall >= 80%"
SHAP_NAMES = S3.FEATURE_VN
KS = (100, 200, 300, 400)


# ----------------------------------------------------------------------------------------------
# Tiện ích đọc bảng
# ----------------------------------------------------------------------------------------------
def num(x) -> float:
    """'0.812 ± 0.003 [0.808; 0.816]' -> 0.812. Số thuần trả về nguyên giá trị."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    m = re.match(r"\s*([-+]?\d+(?:[.,]\d+)?)", str(x))
    return float(m.group(1).replace(",", ".")) if m else float("nan")


def num_sd(x) -> float:
    m = re.search(r"±\s*(\d+(?:[.,]\d+)?)", str(x))
    return float(m.group(1).replace(",", ".")) if m else float("nan")


@st.cache_data(show_spinner=False)
def read_table(name: str):
    p = S3.TABLES / name
    return pd.read_csv(p) if p.exists() else None


def first_table(*names: str):
    for n in names:
        t = read_table(n)
        if t is not None:
            return t
    return None


@st.cache_data(show_spinner=False)
def data_stats():
    """(số hồ sơ, tỉ lệ nhãn dương, nguồn). Đọc từ file dữ liệu; nếu không tìm thấy thì dùng số đã công bố của BRFSS 2015."""
    try:
        df, _ = U.get_data()
        return len(df), float(df[U.TARGET].mean()), "tính từ file dữ liệu"
    except Exception:
        return 253_680, 0.1393, "số liệu mô tả ở Giai đoạn 1"


def dc(x: float, d: int = 1) -> str:
    """Số thực với dấu phẩy thập phân kiểu Việt Nam."""
    return f"{x:.{d}f}".replace(".", ",")


def vn(n) -> str:
    return f"{n:,.0f}".replace(",", ".")


def html(x: str):
    st.markdown(x, unsafe_allow_html=True)


def missing(name: str):
    st.info(f"Chưa có bảng {name} – hãy chạy xong notebook Giai đoạn 3 (mục 3.1–3.4).")


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=10)


def filtered(df: pd.DataFrame, key: str, cols: list[str], max_levels: int = 14) -> pd.DataFrame:
    """Bảng dài: cho lọc theo các cột phân loại (chọn nhiều giá trị) và tìm theo từ khóa."""
    out = df
    cols = [c for c in cols if c in df.columns and df[c].nunique() <= max_levels]
    with st.expander("Bộ lọc bảng", expanded=False):
        cc = st.columns(max(1, min(len(cols), 3)) if cols else 1)
        for i, c in enumerate(cols):
            levels = list(dict.fromkeys(df[c].astype(str)))
            with cc[i % len(cc)]:
                pick = st.multiselect(c, levels, default=levels, key=f"{key}_{c}")
            out = out[out[c].astype(str).isin(pick)]
        q = st.text_input("Tìm theo từ khóa", key=f"{key}_q", placeholder="ví dụ: Recall, BMI, LightGBM")
        if q:
            mask = out.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
            out = out[mask]
    st.caption(f"Hiển thị {len(out)}/{len(df)} dòng.")
    return out


# ----------------------------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------------------------
def model_metrics():
    t = first_table("2_4_test_ranking_summary.csv", "2_final_test_rank_metrics.csv")
    if t is None or "PR-AUC" not in t.columns:
        return None
    d = t[t["Mô hình"] == "LightGBM"] if "Mô hình" in t.columns else t
    if "Kịch bản" in d.columns:
        b = d[d["Kịch bản"].astype(str).str.startswith("B")]
        d = b if not b.empty else d
    if d.empty:
        return None
    r = d.iloc[0]
    return {"pr": num(r["PR-AUC"]), "pr_sd": num_sd(r["PR-AUC"]), "roc": num(r["ROC-AUC"]), "roc_sd": num_sd(r["ROC-AUC"])}


def operating_frame() -> pd.DataFrame | None:
    t = read_table("3_1_operating_points_summary.csv")
    if t is None:
        return None
    d = pd.DataFrame({"op": t["Điểm vận hành"]})
    for src, dst in (("Ngưỡng", "thr"), ("Recall", "recall"), ("Precision", "precision"), ("F1", "f1"),
                     ("FN / 1.000", "fn"), ("FP / 1.000", "fp"), ("Cảnh báo / 1.000", "alerts")):
        d[dst] = t[src].map(num)
    return d


def page_dashboard(system: dict | None):
    n, pos, src = data_stats()
    mm = model_metrics()
    th = (system or {}).get("thresholds")
    html(f'<div class="dss-card">{ui.escape("Kết quả dựa trên dữ liệu khảo sát tự khai (BRFSS 2015, Hoa Kỳ), chỉ mang tính tham khảo, không phải chẩn đoán và không thay thế nhân viên y tế.")}</div>')

    # --- Hàng 1: dữ liệu và mô hình ---------------------------------------------------------
    model_name = (system or {}).get("model_name", "LightGBM")
    html(ui.kpis([
        ("Tổng số hồ sơ", vn(n), "BRFSS 2015, 21 đặc trưng"),
        ("Tỉ lệ nhãn dương", dc(100 * pos, 2) + "%", f"{vn(round(n * pos))} hồ sơ tiền TĐ/TĐ (tự khai)"),
        ("Mô hình đang dùng", model_name, "hiệu chuẩn: " + U.CAL_LABEL.get((system or {}).get("calibration", "none"), "chưa hiệu chuẩn")),
        ("PR-AUC (test)", dc(mm["pr"], 3) if mm else "–", f"± {dc(mm['pr_sd'], 3)} qua 5 seed" if mm else "chưa có bảng"),
        ("ROC-AUC (test)", dc(mm["roc"], 3) if mm else "–", f"± {dc(mm['roc_sd'], 3)} qua 5 seed" if mm else "chưa có bảng"),
    ]))
    st.caption(f"Số hồ sơ và tỉ lệ nhãn dương: {src}. PR-AUC/ROC-AUC tính trên xác suất chưa hiệu chuẩn, kịch bản chia theo hồ sơ (Giai đoạn 2).")

    # --- Hàng 2: điểm vận hành được chọn ----------------------------------------------------
    ops = operating_frame()
    sel_op = DEFAULT_OP
    if ops is not None:
        names = [o for o in OP_ORDER if o in set(ops["op"])] + [o for o in ops["op"] if o not in OP_ORDER]
        sel_op = st.selectbox("Điểm vận hành (ngưỡng quyết định)", names, index=names.index(DEFAULT_OP) if DEFAULT_OP in names else 0,
                              help="Ngưỡng học trên validation; chỉ số dưới đây tính trên tập test, trung bình 5 seed.")
        r = ops[ops["op"] == sel_op].iloc[0]
        html(ui.kpis([
            ("Recall", dc(100 * r.recall) + "%", "tỉ lệ ca dương được phát hiện"),
            ("Precision", dc(100 * r.precision) + "%", "trong số hồ sơ được cảnh báo, tỉ lệ đúng"),
            ("Hồ sơ bỏ sót", f"{r.fn:.0f}", "ca dương không bị cảnh báo / 1.000 hồ sơ"),
            ("Hồ sơ được cảnh báo", f"{r.alerts:.0f}", "/ 1.000 hồ sơ"),
            ("Cảnh báo sai", f"{r.fp:.0f}", "hồ sơ âm bị cảnh báo / 1.000 hồ sơ"),
        ]))
        st.caption(f"Ngưỡng xác suất của điểm vận hành này: {dc(r.thr, 3)}. 'Bỏ sót' và 'cảnh báo' tính trên 1.000 hồ sơ để dễ so sánh giữa các phương án.")
    else:
        missing("3_1_operating_points_summary.csv")

    t1, t2, t3, t4, t5 = st.tabs(["Tổng quan hệ thống", "Ngưỡng và Top-k", "Phân mức 3 tầng", "Yếu tố quan trọng và bỏ sót", "Bảng số liệu"])
    with t1:
        html(ui.steps([("Nhập hồ sơ", "21 thông tin khảo sát"), ("Mô hình dự đoán", "LightGBM cho xác suất"),
                       ("Phân mức và quy tắc", "Thấp / Trung bình / Cao + quy tắc có nguồn"), ("Giải thích và What-if", "SHAP và thử thay đổi lối sống")]))
        p = S3.FIGURES / "3_0_architecture.png"
        if p.exists():
            st.image(str(p), caption="Kiến trúc hệ thống", use_container_width=True)
    with t2:
        threshold_topk_tab(sel_op)
    with t3:
        tier_tab(th)
    with t4:
        importance_and_fn_tab(sel_op, ops)
    with t5:
        tables_tab()


def threshold_topk_tab(sel_op: str):
    t = read_table("3_2_topk_vs_threshold.csv")
    if t is None:
        missing("3_2_topk_vs_threshold.csv")
        return
    d = pd.DataFrame({"name": t["Phương án"], "work": t["Hồ sơ xử lý / 1.000"].map(num), "recall": t["Recall@k"].map(num),
                      "prec": t["Precision@k"].map(num), "fn": t["FN còn lại / 1.000"].map(num), "fp": t["FP / 1.000"].map(num)})
    d["kind"] = np.where(d["name"].str.startswith("Top-k:"), "Top-k theo năng lực",
                         np.where(d["name"].str.startswith("Ngưỡng cố định"), "Ngưỡng cố định", "Top-k bằng số cảnh báo của ngưỡng"))
    kinds = list(dict.fromkeys(d["kind"]))
    show = st.multiselect("Nhóm phương án hiển thị", kinds, default=kinds, key="dash_kinds",
                          help="Top-k 'bằng số cảnh báo của ngưỡng' cho thấy: cùng số hồ sơ được xử lý thì Top-k và ngưỡng gần như cho kết quả như nhau.")
    d = d[d["kind"].isin(show)] if show else d

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        tk = d[d["kind"] == "Top-k theo năng lực"].sort_values("work")
        if not tk.empty:
            ax.plot(tk["work"], tk["recall"], "o-", color=TEAL, lw=2, label="Top-k theo năng lực xử lý")
            for _, r in tk.iterrows():
                ax.annotate(dc(r["recall"], 2), (r["work"], r["recall"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10, color=TEAL)
        fx = d[d["kind"] == "Ngưỡng cố định"]
        ax.scatter(fx["work"], fx["recall"], s=90, marker="s", color=RED, label="Ngưỡng cố định", zorder=3)
        for _, r in fx.iterrows():
            ax.annotate(r["name"].replace("Ngưỡng cố định: ", ""), (r["work"], r["recall"]), textcoords="offset points", xytext=(7, -13), fontsize=9.5)
        mt = d[d["kind"] == "Top-k bằng số cảnh báo của ngưỡng"]
        if not mt.empty:
            ax.scatter(mt["work"], mt["recall"], s=150, marker="o", facecolors="none", edgecolors=INK, linewidths=1.8,
                       label="Top-k cùng số hồ sơ với ngưỡng", zorder=4)
        _style(ax, "Số hồ sơ được xử lý / 1.000 người", "Recall", "Recall theo khối lượng xử lý")
        ax.legend(fontsize=9, loc="lower right")
        show_fig(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        pick = d.sort_values(["work", "kind"])
        y = np.arange(len(pick))
        h = 0.38
        b1 = ax.barh(y - h / 2, pick["fn"], h, color=RED, label="Bỏ sót (FN)")
        b2 = ax.barh(y + h / 2, pick["fp"], h, color=AMBER, label="Cảnh báo sai (FP)")
        ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=9)
        ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace("Ngưỡng cố định: ", "Ngưỡng: ").replace("/1.000", "") for n in pick["name"]], fontsize=9.5)
        ax.invert_yaxis()
        _style(ax, "Số hồ sơ / 1.000 người", "", "Bỏ sót và cảnh báo sai")
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0)
        ax.legend(fontsize=9, loc="lower right")
        show_fig(fig)
    st.info("**Cách đọc:** khi Top-k lấy đúng số hồ sơ mà một ngưỡng xác suất cảnh báo (vòng tròn rỗng trùng với ô vuông đỏ), Recall, bỏ sót và cảnh báo sai gần như bằng nhau. "
            "Top-k không làm LightGBM dự đoán tốt hơn; giá trị của nó là giữ cố định số hồ sơ cần xử lý theo năng lực nguồn lực, "
            "còn ngưỡng cố định cho số hồ sơ thay đổi theo từng tập dữ liệu.")
    st.caption("Trung bình 5 seed trên tập test. Năng lực xử lý (100–400 hồ sơ/1.000 người) là kịch bản giả định.")
    with st.expander("Bảng chi tiết"):
        st.dataframe(filtered(t, "dash_topk", ["Phương án"]), use_container_width=True, hide_index=True)


def tier_tab(th: dict | None):
    t = read_table("3_1_tiers_test_summary.csv")
    if t is None:
        missing("3_1_tiers_test_summary.csv")
        return
    schemes = list(dict.fromkeys(t["Sơ đồ"]))
    default = next((s for s in schemes if th and s.startswith(th["scheme"].split(" ")[0])), schemes[0])
    scheme = st.radio("Sơ đồ phân mức", schemes, index=schemes.index(default), horizontal=True,
                      help="Sơ đồ A đang được hệ thống sử dụng." if th else None)
    d = t[t["Sơ đồ"] == scheme].set_index("Tầng").loc[["Thấp", "Trung bình", "Cao"]]
    cols = [("Tỷ lệ dân số (%)", "Phân bố ba mức cảnh báo\n(% hồ sơ)"), ("Tỷ lệ lớp 1 trong tầng (%)", "Tỉ lệ ca dương trong mỗi mức\n(%)"),
            ("Tỷ lệ ca bệnh được bắt (%)", "Tỉ lệ ca dương thuộc mỗi mức\n(%)")]
    c = st.columns(3)
    for ax_col, (col, title) in zip(c, cols):
        with ax_col:
            vals = d[col].map(num)
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            bars = ax.bar(["▼ Thấp", "◆ Trung bình", "▲ Cao"], vals, color=[TIER_COLORS[i] for i in d.index], edgecolor=INK, linewidth=1)
            ax.bar_label(bars, labels=[dc(v) + "%" for v in vals], padding=3, fontsize=11, fontweight="bold")
            ax.set_ylim(0, max(vals.max() * 1.2, 5))
            ax.tick_params(axis="x", labelsize=9.5)
            _style(ax, "", "", title)
            show_fig(fig)
    if th:
        st.caption(f"Ngưỡng: Thấp < {dc(th['t_low'], 3)} ≤ Trung bình < {dc(th['t_high'], 3)} ≤ Cao. Test, trung bình 5 seed. Mức Thấp vẫn có ca dương: không có nghĩa chắc chắn không mắc bệnh.")
    else:
        st.caption("Test, trung bình 5 seed. Mức Thấp vẫn có ca dương: không có nghĩa chắc chắn không mắc bệnh.")
    with st.expander("Bảng chi tiết"):
        st.dataframe(t, use_container_width=True, hide_index=True)


def importance_and_fn_tab(sel_op: str, ops: pd.DataFrame | None):
    c1, c2 = st.columns(2)
    with c1:
        imp = read_table("3_4_shap_importance.csv")
        if imp is None:
            missing("3_4_shap_importance.csv")
        else:
            n_top = st.slider("Số yếu tố hiển thị", 5, 15, 10, key="dash_topn")
            d = imp.sort_values("mean(|SHAP|)", ascending=False).head(n_top).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.4, 0.46 * n_top + 1.6))
            bars = ax.barh([SHAP_NAMES.get(f, f) for f in d["Đặc trưng"]], d["mean(|SHAP|)"], color=TEAL)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
            ax.margins(x=0.15)
            _style(ax, "mean(|SHAP|), log-odds của mô hình thô", "", "Yếu tố quan trọng nhất (toàn cục)")
            ax.grid(axis="x", alpha=0.25)
            ax.grid(axis="y", alpha=0)
            show_fig(fig)
            st.caption("SHAP mô tả cách mô hình dùng biến, không phải quan hệ nhân quả.")
    with c2:
        fn = read_table("3_3_fn_by_group.csv")
        if fn is None:
            missing("3_3_fn_by_group.csv")
            return
        ops_avail = list(dict.fromkeys(fn["Điểm vận hành"]))
        op = sel_op if sel_op in ops_avail else (DEFAULT_OP if DEFAULT_OP in ops_avail else ops_avail[0])
        if op != sel_op:
            st.caption(f"Phân tích bỏ sót theo nhóm chỉ có cho: {', '.join(ops_avail)}. Đang hiển thị '{op}'.")
        groups = list(dict.fromkeys(fn["Nhóm"]))
        pick = st.multiselect("Nhóm đối tượng", groups, default=groups, key="dash_fn_groups")
        d = fn[(fn["Điểm vận hành"] == op) & (fn["Nhóm"].isin(pick))].copy()
        if d.empty:
            st.info("Chọn ít nhất một nhóm.")
        else:
            d["label"] = d["Nhóm"] + ": " + d["Mức"].astype(str)
            d = d.sort_values(["Nhóm", "FNR"], ascending=[False, True])
            fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(d) + 1.8))
            low_n = d["Số lớp 1"] < 100
            err = [100 * (d["FNR"] - d["FNR CI95 dưới"]).clip(lower=0), 100 * (d["FNR CI95 trên"] - d["FNR"]).clip(lower=0)]
            bars = ax.barh(d["label"], 100 * d["FNR"], xerr=err, color=[AMBER if x else RED for x in low_n], ecolor=INK, capsize=3)
            ax.bar_label(bars, labels=[dc(100 * v) + "%" for v in d["FNR"]], padding=24, fontsize=9.5)
            if ops is not None and op in set(ops["op"]):
                overall = 100 * (1 - float(ops.loc[ops["op"] == op, "recall"].iloc[0]))
                ax.axvline(overall, color=INK, ls="--", lw=1.4, label=f"Bỏ sót chung: {dc(overall)}%")
                ax.legend(fontsize=9, loc="lower right")
            ax.set_xlim(0, max(100 * d["FNR CI95 trên"].max() * 1.18, 10))
            _style(ax, "Tỉ lệ bỏ sót (FNR, %) và khoảng tin cậy 95%", "", f"Bỏ sót theo nhóm – {op}")
            ax.grid(axis="x", alpha=0.25)
            ax.grid(axis="y", alpha=0)
            show_fig(fig)
            st.caption("Thanh vàng: nhóm có dưới 100 ca dương, diễn giải thận trọng. Kết quả trên tập test, seed 42.")


def tables_tab():
    catalog = {
        "Điểm vận hành (test)": "3_1_operating_points_summary.csv",
        "Phân mức 3 tầng (test)": "3_1_tiers_test_summary.csv",
        "Top-k so với ngưỡng cố định": "3_2_topk_vs_threshold.csv",
        "Bỏ sót theo nhóm": "3_3_fn_by_group.csv",
        "Độ quan trọng SHAP": "3_4_shap_importance.csv",
        "So sánh cấu hình A/B/C": "3_7_configs_ABC.csv",
    }
    name = st.selectbox("Chọn bảng", list(catalog), key="dash_table_pick")
    t = read_table(catalog[name])
    if t is None:
        missing(catalog[name])
        return
    st.dataframe(filtered(t, "dash_tbl_" + catalog[name], ["Điểm vận hành", "Nhóm", "Sơ đồ", "Tầng", "Cấu hình", "Phương án"]),
                 use_container_width=True, hide_index=True)
    st.download_button("Tải bảng này (CSV)", t.to_csv(index=False).encode("utf-8-sig"), catalog[name], "text/csv")


# ----------------------------------------------------------------------------------------------
# Capacity Planning: đánh đổi khi năng lực xử lý thay đổi
# ----------------------------------------------------------------------------------------------
def capacity_frame() -> pd.DataFrame | None:
    t = read_table("3_2_topk_vs_threshold.csv")
    if t is None:
        return None
    rows = []
    for k in KS:
        r = t[t["Phương án"] == f"Top-k: {k}/1.000"]
        if r.empty:
            continue
        r = r.iloc[0]
        work, fp, fn = num(r["Hồ sơ xử lý / 1.000"]), num(r["FP / 1.000"]), num(r["FN còn lại / 1.000"])
        rows.append({"k": k, "recall": num(r["Recall@k"]), "precision": num(r["Precision@k"]), "work": work, "tp": work - fp, "fn": fn, "fp": fp})
    return pd.DataFrame(rows) if rows else None


def capacity_tradeoff(k_sel: int):
    d = capacity_frame()
    st.subheader("Đánh đổi khi thay đổi năng lực xử lý")
    if d is None:
        missing("3_2_topk_vs_threshold.csv")
        return
    st.caption("Cùng 1.000 hồ sơ, chỉ thay đổi số hồ sơ có thể tiếp nhận. Kết quả tham chiếu trên tập test, trung bình 5 seed. Năng lực xử lý là kịch bản giả định.")

    d = d.sort_values("k").reset_index(drop=True)
    d["d_tp"] = d["tp"].diff()
    d["d_fp"] = d["fp"].diff()
    d["d_rec"] = d["recall"].diff()
    d["fp_per_tp"] = d["d_fp"] / d["d_tp"]

    tab = pd.DataFrame({
        "Năng lực xử lý (hồ sơ/1.000)": d["k"],
        "Recall (%)": (100 * d["recall"]).round(1),
        "Precision (%)": (100 * d["precision"]).round(1),
        "Ca dương phát hiện": d["tp"].round(1),
        "Bỏ sót (FN)": d["fn"].round(1),
        "Cảnh báo sai (FP)": d["fp"].round(1),
        "Recall tăng thêm (điểm %)": (100 * d["d_rec"]).round(1),
        "Ca dương phát hiện thêm": d["d_tp"].round(1),
        "Cảnh báo sai thêm": d["d_fp"].round(1),
        "Cảnh báo sai thêm cho mỗi ca phát hiện thêm": d["fp_per_tp"].round(1),
    })
    base = d[d["k"] == KS[0]].iloc[0] if (d["k"] == KS[0]).any() else d.iloc[0]
    cur = d[d["k"] == k_sel].iloc[0] if (d["k"] == k_sel).any() else d.iloc[0]
    html(ui.kpis([
        ("Recall", dc(100 * cur.recall) + "%", f"so với {dc(100 * base.recall)}% khi xử lý {KS[0]}"),
        ("Ca dương bị bỏ sót", f"{cur.fn:.0f}", f"/ 1.000 hồ sơ (khi xử lý {KS[0]}: {base.fn:.0f})"),
        ("Cảnh báo sai", f"{cur.fp:.0f}", f"/ 1.000 hồ sơ (khi xử lý {KS[0]}: {base.fp:.0f})"),
        ("Precision", dc(100 * cur.precision) + "%", "giảm dần khi mở rộng danh sách"),
    ]))

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.plot(d["k"], 100 * d["recall"], "o-", color=TEAL, lw=2.4, ms=8)
        for _, r in d.iterrows():
            ax.annotate(dc(100 * r["recall"]) + "%", (r["k"], 100 * r["recall"]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10.5, fontweight="bold")
        ax.axvline(k_sel, color=RED, ls=":", lw=1.6)
        ax.set_xticks(list(d["k"]))
        ax.set_ylim(0, max(100, 100 * d["recall"].max() * 1.1))
        _style(ax, "Năng lực xử lý (hồ sơ / 1.000 người)", "Recall (%)", "Recall tăng chậm dần")
        show_fig(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        x, w = np.arange(len(d)), 0.38
        b1 = ax.bar(x - w / 2, d["fn"], w, color=RED, label="Bỏ sót (FN)")
        b2 = ax.bar(x + w / 2, d["fp"], w, color=AMBER, label="Cảnh báo sai (FP)")
        ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=10)
        ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(k)) for k in d["k"]])
        ax.set_ylim(0, max(d["fn"].max(), d["fp"].max()) * 1.18)
        _style(ax, "Năng lực xử lý (hồ sơ / 1.000 người)", "Số hồ sơ / 1.000 người", "Bỏ sót giảm, cảnh báo sai tăng")
        ax.legend(fontsize=9.5)
        show_fig(fig)

    st.dataframe(tab, use_container_width=True, hide_index=True)
    if k_sel != KS[0] and (d["k"] == k_sel).any():
        i = int(d.index[d["k"] == k_sel][0])
        prev = d.iloc[i - 1]
        st.markdown(
            f"**Đọc kết quả:** nâng năng lực từ {int(prev.k)} lên {int(cur.k)} hồ sơ/1.000 người giúp phát hiện thêm khoảng **{dc(cur.tp - prev.tp, 0)}** ca dương "
            f"(Recall {dc(100 * prev.recall)}% → {dc(100 * cur.recall)}%, số bỏ sót {dc(prev.fn, 0)} → {dc(cur.fn, 0)}), nhưng cũng thêm khoảng **{dc(cur.fp - prev.fp, 0)}** cảnh báo sai "
            f"(khoảng {dc((cur.fp - prev.fp) / max(cur.tp - prev.tp, 1e-9))} cảnh báo sai cho mỗi ca phát hiện thêm).")
    else:
        st.markdown(f"**Đọc kết quả:** ở mức {KS[0]} hồ sơ/1.000 người, số hồ sơ bỏ sót còn nhiều; tăng năng lực sẽ giảm bỏ sót nhưng kéo theo nhiều cảnh báo sai hơn.")
    st.caption("Lợi ích thêm của mỗi lần mở rộng giảm dần, còn chi phí cảnh báo sai tăng dần. Đây là cơ sở để chọn năng lực phù hợp; mức chấp nhận được do đơn vị triển khai quyết định.")


# ----------------------------------------------------------------------------------------------
# What-if: bảng trước/sau, biểu đồ, giải thích ngắn
# ----------------------------------------------------------------------------------------------
def _show_value(k: str, v):
    if k == "BMI":
        return f"{float(v):.1f}".replace(".", ",")
    if k in S3.VALUE_LABELS:
        return S3.VALUE_LABELS[k].get(v, v)
    if k in S3.BINARY_FEATURES:
        return S3.YESNO.get(int(v), v)
    return v


def whatif_view(profile: dict, changes: dict, out: dict, system: dict):
    b, a, delta, th = out["before"], out["after"], out["delta_p"], system["thresholds"]
    html(ui.compare(b, a, delta, th))

    # --- Bảng so sánh trước và sau ---------------------------------------------------------
    rows = [("Xác suất dự đoán", f"{b['p']:.3f}", f"{a['p']:.3f}", f"{delta:+.3f}"),
            ("Mức nguy cơ", f"{ui.TIER_ICON[b['tier']]}{b['tier_name']}", f"{ui.TIER_ICON[a['tier']]}{a['tier_name']}", "giữ nguyên" if b["tier"] == a["tier"] else "đổi mức"),
            ("Mức ưu tiên hỗ trợ", b["priority_name"], a["priority_name"], "giữ nguyên" if b["priority_name"] == a["priority_name"] else "đổi"),
            ("Số quy tắc được kích hoạt", str(len(b["rules"])), str(len(a["rules"])), f"{len(a['rules']) - len(b['rules']):+d}")]
    for k, v in changes.items():
        rows.append((f"Thay đổi: {S3.FEATURE_VN[k]}", str(_show_value(k, profile[k])), str(_show_value(k, v)), ""))
    st.dataframe(pd.DataFrame(rows, columns=["Chỉ số", "Trước", "Sau", "Chênh lệch"]).astype(str), use_container_width=True, hide_index=True)

    # --- Biểu đồ -----------------------------------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        bars = ax.bar(["Trước", "Sau"], [b["p"], a["p"]], color=[MUTED, TEAL], width=0.5)
        ax.bar_label(bars, labels=[f"{b['p']:.3f}", f"{a['p']:.3f}"], padding=3, fontsize=12, fontweight="bold")
        ax.axhline(th["t_low"], color=AMBER, ls="--", lw=1.5)
        ax.axhline(th["t_high"], color=RED, ls="--", lw=1.5)
        xmax = ax.get_xlim()[1]
        ax.text(xmax, th["t_low"], f" Thấp | Trung bình\n {th['t_low']:.3f}", va="center", fontsize=8.5, clip_on=False)
        ax.text(xmax, th["t_high"], f" Trung bình | Cao\n {th['t_high']:.3f}", va="center", fontsize=8.5, clip_on=False)
        ax.set_ylim(0, max(b["p"], a["p"], th["t_high"]) * 1.25)
        _style(ax, "", "Xác suất dự đoán", "Xác suất trước và sau thay đổi")
        show_fig(fig)
    with c2:
        w = out["why"].copy()
        w["Tên"] = w["Đặc trưng"].map(lambda f: S3.FEATURE_VN.get(f, f))
        w = w.iloc[::-1]
        vals = w["Thay đổi SHAP (log-odds thô)"].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4.6))
        bars = ax.barh(w["Tên"], vals, color=[RED if v > 0 else BLUE for v in vals])
        ax.bar_label(bars, labels=[f"{v:+.2f} ({'tăng' if v > 0 else 'giảm'})" for v in vals], padding=3, fontsize=9.5)
        ax.axvline(0, color="grey", lw=0.8)
        ax.margins(x=0.35)
        _style(ax, "Thay đổi đóng góp SHAP (log-odds mô hình thô)", "", "Yếu tố làm kết quả đổi")
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0)
        show_fig(fig)

    # --- Giải thích ngắn ---------------------------------------------------------------------
    top = out["why"].iloc[0]
    top_name = S3.FEATURE_VN.get(top["Đặc trưng"], top["Đặc trưng"])
    direction = "giảm" if delta < 0 else "tăng" if delta > 0 else "không thay đổi"
    tier_txt = (f"Mức nguy cơ vẫn là **{a['tier_name']}**." if b["tier"] == a["tier"]
                else f"Mức nguy cơ chuyển từ **{b['tier_name']}** sang **{a['tier_name']}**.")
    st.markdown(f"**Giải thích ngắn:** với các thay đổi đã chọn, xác suất mô hình dự đoán {direction} (từ {b['p']:.3f} thành {a['p']:.3f}, chênh {delta:+.3f}). "
                f"{tier_txt} Yếu tố làm đầu ra thay đổi nhiều nhất là **{top_name}**. "
                "Mức thay đổi này là phản ứng của mô hình, không phải mức giảm nguy cơ thực tế.")
    st.warning(S3.WHATIF_DISCLAIMER)

    t1, t2 = st.tabs(["Chi tiết SHAP", "Lưu ý độ tin cậy"])
    with t1:
        why = out["why"].copy()
        why.insert(1, "Tên", why["Đặc trưng"].map(lambda f: S3.FEATURE_VN.get(f, f)))
        st.dataframe(why, use_container_width=True, hide_index=True)
    with t2:
        for f in out["flags"]:
            st.warning(f)
        if not out["flags"]:
            st.success("Không có cảnh báo về độ hợp lý của thay đổi.")
        st.caption(f"Số hồ sơ huấn luyện tương tự hồ sơ sau thay đổi: {out['support']}.")


# ----------------------------------------------------------------------------------------------
# Model Evaluation: nhóm theo tab, có bộ lọc
# ----------------------------------------------------------------------------------------------
def first_image(*names: str):
    for n in names:
        p = S3.FIGURES / n
        if p.exists():
            return p
    return None


def page_eval():
    """Mỗi nhóm: (bảng: tiêu đề, [tên file, có bản _quick dự phòng], cột lọc) và (hình: chính, dự phòng, chú thích)."""
    groups = {
        "Hiệu chuẩn và ngưỡng": ([("Hiệu chuẩn (validation, 5 seed)", ["3_0_calibration_validation_summary.csv"], ["Hiệu chuẩn", "Phương pháp"]),
                                  ("Điểm vận hành (test)", ["3_1_operating_points_summary.csv"], ["Điểm vận hành"])],
                                 [("2_6_calibration_curves.png", "2_6_calibration_curves_quick.png", "Đường hiệu chuẩn: chưa hiệu chuẩn so với Platt và Isotonic (test, seed đầu tiên)"),
                                  ("2_7_threshold_tradeoff.png", "2_7_threshold_tradeoff_quick.png", "Đánh đổi Recall – Precision – tỷ lệ cảnh báo theo ngưỡng")]),
        "Top-k và nhóm bị bỏ sót": ([("Top-k so với ngưỡng cố định", ["3_2_topk_vs_threshold.csv"], ["Phương án"]),
                                     ("Bỏ sót theo nhóm", ["3_3_fn_by_group.csv"], ["Điểm vận hành", "Nhóm"])],
                                    [("3_2_topk_vs_threshold.png", None, "Top-k so với ngưỡng cố định"),
                                     ("3_3_fn_by_group.png", None, "Bỏ sót theo nhóm đối tượng")]),
        "So sánh cấu hình": ([("So sánh cấu hình A/B/C", ["3_7_configs_ABC.csv"], ["Cấu hình", "Phương án"]),
                              ("Chất lượng hệ hỗ trợ quyết định (D)", ["3_7_system_quality_D.csv"], [])], []),
        "Ablation đặc trưng": ([("Ablation: bỏ dần các biến yếu (validation, trung bình các seed)",
                                 ["2_9_feature_ablation_summary.csv", "2_9_feature_ablation_summary_quick.csv"], ["Phương án"]),
                                ("Chênh lệch ghép cặp so với bản đủ 21 biến",
                                 ["2_9_feature_ablation_paired_diff.csv", "2_9_feature_ablation_paired_diff_quick.csv"], ["Phương án"]),
                                ("Độ quan trọng hoán vị (permutation importance)",
                                 ["2_9_permutation_importance.csv", "2_9_permutation_importance_quick.csv"], [])],
                               [("2_9_feature_ablation.png", "2_9_feature_ablation_quick.png", "Ablation đặc trưng")]),
        "So sánh thuật toán": ([], []),
    }
    notes = {"Ablation đặc trưng": "Mô hình chính vẫn dùng đủ 21 biến theo yêu cầu; mục này chỉ là bằng chứng để bàn về bản rút gọn.",
             "So sánh cấu hình": "Top-k, SHAP, quy tắc và What-if không làm tăng PR-AUC; chúng được đánh giá qua chi phí vận hành, tính đúng đắn và tính nhất quán."}
    for tab, (gname, (items, imgs)) in zip(st.tabs(list(groups)), groups.items()):
        with tab:
            if gname in notes:
                st.caption(notes[gname])
            for title, names, fcols in items:
                t = first_table(*names)
                if t is None:
                    st.info(f"Chưa có bảng {names[0]}.")
                    continue
                st.subheader(title)
                st.dataframe(filtered(t, "ev_" + names[0], fcols), use_container_width=True, hide_index=True)
            for main, alt, cap in imgs:
                p = first_image(*(n for n in (main, alt) if n))
                if p:
                    st.image(str(p), caption=cap, use_container_width=True)
            if gname == "So sánh thuật toán":
                algorithm_tab()


def algorithm_tab():
    t = first_table("2_4_test_ranking_summary.csv", "2_final_test_rank_metrics.csv")
    if t is None:
        st.info("Chưa có bảng 2_4_test_ranking_summary.csv (Giai đoạn 2).")
        return
    d = t
    if "Kịch bản" in d.columns:
        sc = st.radio("Kịch bản chia dữ liệu", list(dict.fromkeys(d["Kịch bản"])), horizontal=True, key="ev_scen")
        d = d[d["Kịch bản"] == sc]
    d = d.assign(pr=d["PR-AUC"].map(num), roc=d["ROC-AUC"].map(num))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(d) + 1.8))
    y, h = np.arange(len(d)), 0.38
    b1 = ax.barh(y - h / 2, d["pr"], h, color=TEAL, label="PR-AUC")
    b2 = ax.barh(y + h / 2, d["roc"], h, color=AQUA, label="ROC-AUC")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9.5)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Mô hình"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    _style(ax, "Điểm (tập test, trung bình 5 seed, xác suất chưa hiệu chuẩn)", "", "So sánh thuật toán")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0)
    ax.legend(fontsize=9, loc="lower right")
    show_fig(fig)
    st.dataframe(filtered(t, "ev_algo", ["Kịch bản", "Mô hình", "Chiến lược"]), use_container_width=True, hide_index=True)
