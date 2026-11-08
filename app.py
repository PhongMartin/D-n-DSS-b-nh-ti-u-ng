"""Ứng dụng Streamlit – Hệ hỗ trợ quyết định đánh giá nguy cơ tiền tiểu đường/tiểu đường (INS305).

Chạy:  streamlit run app.py        (cần đã chạy xong notebook Giai đoạn 3 để có models/Phase3_system.joblib)
Lưu ý: kết quả chỉ mang tính tham khảo từ dữ liệu khảo sát tự khai (BRFSS 2015); không phải chẩn đoán y khoa.
"""
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dss_stage3 as S3
import dss_utils as U
import ui
import export_report as ER
import dashboard as DB

st.set_page_config(page_title="DSS nguy cơ tiền tiểu đường/tiểu đường", page_icon="🫀", layout="wide")
LIMIT_TEXT = ("Kết quả dựa trên dữ liệu khảo sát tự khai (BRFSS 2015, Hoa Kỳ), chỉ mang tính tham khảo, không phải chẩn đoán và không thay thế nhân viên y tế. "
              "Mức nguy cơ thấp không có nghĩa chắc chắn không mắc bệnh.")
# Giao diện: bảng màu "Healthcare" (nền sáng, chữ #164E63, nút #0E7490 để chữ trắng đạt tương phản ≥ 4,5:1);
# mức nguy cơ luôn có biểu tượng + chữ, không chỉ dựa vào màu.
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Be+Vietnam+Pro:wght@400;500;600&display=swap');
:root { --ink:#0A2A33; --teal:#0E5A66; --aqua:#19A7B8; --paper:#F3F7F8; --line:#D5E3E6; --muted:#52666B; }
html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif; font-size: 16px; line-height: 1.6; color: var(--ink); }
[data-testid="stAppViewContainer"] { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; max-width: 1180px; }
h2, h3, h4 { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; font-weight: 800; color: var(--ink); letter-spacing: -0.02em; }
section[data-testid="stSidebar"] { background: #E8F1F3; border-right: 1px solid var(--line); }
.dss-brand { display:flex; align-items:center; gap:10px; padding: 4px 0 14px; }
.dss-brand b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.25rem; color:var(--ink); line-height:1.1; }
.dss-brand small { display:block; color:var(--muted); font-size:.78rem; font-weight:400; }
div[data-testid="stRadio"] label { padding: 8px 12px; border-radius: 10px; width: 100%; transition: background .15s; }
div[data-testid="stRadio"] label:hover { background: #D2E6EA; }
div[data-testid="stRadio"] label:has(input:checked) { background: var(--ink); }
div[data-testid="stRadio"] label:has(input:checked) p { color: #FFFFFF; font-weight: 600; }
div[data-testid="stRadio"] label > div:first-child { display: none; }
.dss-hero { position: relative; overflow: hidden; background: radial-gradient(circle at 88% 0%, rgba(25,167,184,.55), transparent 45%), linear-gradient(120deg, #0A2A33 0%, #0E5A66 100%); box-shadow: 0 14px 40px rgba(10,42,51,.25); border-radius: 20px; padding: 34px 40px 36px; margin-bottom: 22px; }
.dss-hero svg { position: absolute; inset: 0; width: 100%; height: 100%; opacity: .9; }
.dss-hero svg path { stroke-dasharray: 1400; stroke-dashoffset: 1400; animation: trace 2.2s ease-out forwards; }
@keyframes trace { to { stroke-dashoffset: 0; } }
.dss-hero h1 { position: relative; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.6rem; letter-spacing: -0.03em; color: #FFFFFF; margin: 0; padding: 0; line-height: 1.1; }
.dss-hero p { position: relative; color: #BFE3E8 !important; margin: 10px 0 0; max-width: 62ch; font-size: 1.02rem; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; }
div[data-testid="stMetricValue"] { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; }
.dss-card { background: #FFFFFF; border: 1px solid var(--line); border-radius: 16px; padding: 18px 22px; margin-bottom: 14px; }
.dss-result { display: flex; flex-wrap: wrap; gap: 8px 28px; align-items: center; background: #FFFFFF; border: 1px solid var(--line); border-radius: 20px; padding: 18px 26px; margin-bottom: 14px; }
.dss-result .gauge { flex: 0 0 250px; max-width: 100%; }
.dss-result .info { flex: 1 1 260px; }
.dss-result .needle { transform-origin: 150px 150px; animation: sweep 1.1s cubic-bezier(.2,.8,.2,1); }
@keyframes sweep { from { transform: rotate(-80deg); } to { transform: rotate(0); } }
.dss-big { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.1rem; line-height: 1; margin: 0 0 4px; }
.dss-badge { display: inline-block; font-weight: 600; font-size: 1.05rem; padding: 5px 14px; border-radius: 999px; border: 2px solid; margin-bottom: 8px; }
.dss-low { color: #14532D; background: #DCFCE7; border-color: #14532D; }
.dss-medium { color: #78350F; background: #FEF3C7; border-color: #78350F; }
.dss-high { color: #7F1D1D; background: #FEE2E2; border-color: #7F1D1D; }
.dss-note { color: var(--muted); font-size: 0.9rem; }
button[kind="primary"], .stDownloadButton button { background: var(--teal); color: #FFFFFF; border: 0; border-radius: 10px; }
details { border-radius: 12px !important; border-color: var(--line) !important; background: #FFFFFF; }
.dss-chips { position: relative; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.dss-chips span { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.28); color: #FFFFFF; padding: 3px 14px; border-radius: 999px; font-size: .85rem; }
div[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 6px 24px rgba(10,42,51,.06); }
div[data-testid="stExpander"] summary { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.05rem; }
div[data-testid="stSelectbox"] label p, div[data-testid="stNumberInput"] label p, div[data-testid="stToggle"] label p { font-size: .85rem; color: var(--muted); }
div[data-testid="stMetric"] { box-shadow: 0 6px 20px rgba(10,42,51,.06); border-top: 4px solid var(--aqua); }
:focus-visible { outline: 3px solid var(--aqua) !important; outline-offset: 2px; }
button, [role="button"], label { min-height: 44px; }
@media (max-width: 640px) { .dss-hero { padding: 24px 22px; } .dss-hero h1 { font-size: 1.9rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } .dss-hero svg path { stroke-dashoffset: 0; } }
</style>
"""
TIER_ICON = {"low": "▼ ", "medium": "◆ ", "high": "▲ "}
C_POS, C_NEG = "#B91C1C", "#0369A1"      # tăng / giảm đầu ra (kèm nhãn số trên biểu đồ, không chỉ màu)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#F3F7F8", "axes.facecolor": "#F3F7F8", "text.color": "#0A2A33", "axes.labelcolor": "#0A2A33",
                     "xtick.color": "#0A2A33", "ytick.color": "#0A2A33", "axes.edgecolor": "#8FA9AF"})
PAGES = ["Dashboard", "Risk Assessment", "Explainable AI", "Decision Support", "What-if Analysis",
         "Capacity Planning", "Model Evaluation", "Export Results"]
st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(ui.EXTRA_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_system():
    return S3.load_system()


@st.cache_data
def get_rules():
    return S3.load_rules()


def read_table(name: str):
    p = S3.TABLES / name
    return pd.read_csv(p) if p.exists() else None


def show_image(name: str, caption: str = ""):
    p = S3.FIGURES / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Chưa có biểu đồ {name} – hãy chạy notebook Giai đoạn 3.")


def label_select(label, options: dict, default, key):
    keys = list(options)
    return st.selectbox(label, keys, index=keys.index(default), format_func=lambda k: options[k], key=key)


def profile_form() -> dict:
    """Biểu mẫu gọn: lưới 4 cột, công tắc bật/tắt cho các yếu tố có/không."""
    d = S3._neutral()
    p = {}
    with st.expander("Hồ sơ đánh giá: chỉnh thông tin tại đây", expanded=True):
        st.markdown("**Thông tin chung**")
        c = st.columns(4)
        with c[0]: p["Age"] = label_select("Nhóm tuổi", S3.AGE_LABELS, d["Age"] + 4, "Age")
        with c[1]: p["Sex"] = label_select("Giới tính", S3.SEX_LABELS, 0, "Sex")
        with c[2]: p["GenHlth"] = label_select("Sức khỏe tự đánh giá", S3.GENHLTH_LABELS, 3, "GenHlth")
        with c[3]: p["BMI"] = st.number_input("BMI (kg/m²)", 12.0, 98.0, 27.0, 0.1, key="BMI")
        c = st.columns(4)
        with c[0]: p["Education"] = label_select("Học vấn", S3.EDUCATION_LABELS, 5, "Education")
        with c[1]: p["Income"] = label_select("Thu nhập hộ gia đình", S3.INCOME_LABELS, 6, "Income")
        with c[2]: p["MentHlth"] = int(st.number_input(S3.FEATURE_VN["MentHlth"], 0, 30, 0, key="MentHlth"))
        with c[3]: p["PhysHlth"] = int(st.number_input(S3.FEATURE_VN["PhysHlth"], 0, 30, 0, key="PhysHlth"))
        st.markdown("**Bệnh nền và thói quen** (bật nếu có)")
        cols = st.columns(4)
        for i, f in enumerate(S3.BINARY_FEATURES):
            with cols[i % 4]:
                p[f] = int(st.toggle(S3.FEATURE_VN[f], value=bool(d[f]), key="t_" + f))
    return {f: (float(p[f]) if f == "BMI" else int(p[f])) for f in S3.FEATURES}


NAV_ICON = {"Dashboard": "▦", "Risk Assessment": "◔", "Explainable AI": "◈", "Decision Support": "✚",
            "What-if Analysis": "⇄", "Capacity Planning": "☰", "Model Evaluation": "▤", "Export Results": "⇩"}
HERO_SUB = {
    "Dashboard": "Tổng quan hệ thống, các điểm vận hành và những yếu tố mô hình dựa vào nhiều nhất.",
    "Risk Assessment": "Chỉnh hồ sơ trong khung bên dưới để xem mức nguy cơ, đồng hồ và yếu tố ảnh hưởng.",
    "Explainable AI": "Yếu tố nào đang đẩy nguy cơ của hồ sơ này lên hoặc xuống.",
    "Decision Support": "Quy tắc được kích hoạt và hành động tham khảo tương ứng.",
    "What-if Analysis": "Thử thay đổi lối sống và xem mô hình phản ứng ra sao.",
    "Capacity Planning": "Xếp hạng nhiều hồ sơ và chọn Top-k theo năng lực xử lý (kịch bản giả định).",
    "Model Evaluation": "Hiệu chuẩn, điểm vận hành, Top-k, cấu hình A–D, ablation đặc trưng và so sánh thuật toán.",
    "Export Results": "Tải kết quả đánh giá dưới dạng báo cáo HTML/PDF, Excel nhiều trang hoặc CSV.",
}
ECG = ('<svg viewBox="0 0 600 100" preserveAspectRatio="none" aria-hidden="true"><path fill="none" stroke="#19A7B8" stroke-width="2" vector-effect="non-scaling-stroke" '
       'd="M0 62 H90 L105 62 L115 40 L128 88 L140 14 L152 76 L160 62 H300 L315 62 L325 44 L338 86 L350 26 L362 70 L370 62 H600"/></svg>')


def hero(title: str):
    st.markdown(f'<div class="dss-hero">{ECG}<div class="dss-eyebrow">Hệ hỗ trợ quyết định · tiền tiểu đường / tiểu đường</div><h1>{title}</h1><p>{HERO_SUB.get(title, "")}</p><div class="dss-chips"><span>253.680 hồ sơ</span><span>21 đặc trưng</span><span>LightGBM + SHAP</span><span>Quy tắc có nguồn</span><span>Không thay thế nhân viên y tế</span></div></div>', unsafe_allow_html=True)


def html(x: str):
    st.markdown(x, unsafe_allow_html=True)


def footer():
    html(ui.footer(LIMIT_TEXT))


def tier_banner(res: dict, system: dict = None):
    """Đồng hồ nguy cơ + mức ưu tiên; mức thấp luôn kèm lưu ý giới hạn. Mức nguy cơ luôn có biểu tượng + chữ."""
    html(ui.result_card(res, (system or {}).get("thresholds")))
    (st.warning if res["tier"] != "low" else st.info)(res["disclaimer"])


def page_dashboard():
    try:
        system = get_system()
    except Exception:
        system = None          # vẫn hiển thị phần đọc từ bảng CSV
    DB.page_dashboard(system)
    footer()


def page_risk(system, rules, profile):
    res = S3.assess(system, profile, rules)
    th = system["thresholds"]
    tier_banner(res, system)
    html(ui.kpis([("Xác suất dự đoán", f"{res['p']:.3f}", "đầu ra của mô hình"), ("Quy tắc kích hoạt", str(len(res["rules"])), "xem chức năng Decision Support"),
                  ("Thời gian xử lý", f"{res['elapsed_ms']:.0f} ms", "một hồ sơ"), ("Hiệu chuẩn", U.CAL_LABEL[system["calibration"]], f"sơ đồ ngưỡng {th['scheme']}")]))
    c = res["contributions"]
    html(ui.factor_bars(list(zip(c["Tên"], c["SHAP (log-odds thô)"])), top=5))
    st.caption(f"Ngưỡng phân mức: Thấp < {th['t_low']:.3f} ≤ Trung bình < {th['t_high']:.3f} ≤ Cao.")
    footer()
    return res


def page_xai(system, rules, profile):
    res = S3.assess(system, profile, rules, top=8)
    tier_banner(res, system)
    t1, t2, t3 = st.tabs(["Hồ sơ này", "Toàn cục", "Theo nhóm đối tượng"])
    with t1:
        c = res["contributions"]
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = list(c["SHAP (log-odds thô)"][::-1])
        bars = ax.barh(c["Tên"][::-1], vals, color=[C_POS if v > 0 else C_NEG for v in vals])
        ax.bar_label(bars, labels=[f"{v:+.2f} ({'tăng' if v > 0 else 'giảm'})" for v in vals], padding=3, fontsize=9)
        ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("Đóng góp SHAP (log-odds của mô hình thô)"); ax.margins(x=0.25)
        st.pyplot(fig)
        st.caption(f"Thang đo: log-odds đầu ra thô của LightGBM. Nền = {res['shap_base_logodds']:.2f} (trung bình log-odds, không phải tỉ lệ nền); "
                   f"tổng = {res['shap_total_logodds']:.2f}. Xác suất hiển thị ({res['p']:.3f}) là xác suất sau cấu hình hiệu chuẩn đã chọn. SHAP mô tả cách mô hình dùng biến, không phải quan hệ nhân quả.")
        st.dataframe(c, use_container_width=True)
    with t2:
        show_image("3_4_shap_beeswarm.png", "Toàn cục")
    with t3:
        show_image("3_4_shap_by_group.png", "Theo nhóm đối tượng")
    footer()


def page_decision(system, rules, profile):
    res = S3.assess(system, profile, rules)
    tier_banner(res, system)
    st.subheader("Quy tắc được kích hoạt")
    if not res["rules"]:
        st.info("Không có quy tắc nào được kích hoạt cho hồ sơ này.")
    for r in sorted(res["rules"], key=lambda r: -r["priority"]):
        html(ui.rule_card(r, S3.PRIORITY_NAMES))
    with st.expander("Xem toàn bộ bộ quy tắc của hệ thống"):
        allr = pd.DataFrame([{"Mã": r["id"], "Tên": r["name"], "Loại": "Chuyên môn" if r["type"] == "guideline" else "Vận hành (nhóm xây dựng)",
                              "Ưu tiên": S3.PRIORITY_NAMES[r["priority"]], "Nguồn": r["source"],
                              "Đang kích hoạt": "Có" if r["id"] in {x["id"] for x in res["rules"]} else "Không"} for r in rules["rules"]])
        st.dataframe(DB.filtered(allr, "dec_rules", ["Loại", "Ưu tiên", "Đang kích hoạt"]), use_container_width=True, hide_index=True)
    footer()
    return res


def page_whatif(system, rules, profile):
    st.warning(S3.WHATIF_DISCLAIMER)
    st.caption("Chỉ cho phép: giảm BMI; bắt đầu vận động; ăn rau; ăn trái cây. Không đổi tuổi, giới, đột quỵ, bệnh tim, hút thuốc (Smoker)...")
    changes = {}
    if st.checkbox("Giảm BMI"):
        lo = max(15.0, float(profile["BMI"]) * 0.85)
        changes["BMI"] = st.slider("BMI mới", float(round(lo, 1)), float(profile["BMI"]), float(profile["BMI"]), 0.1)
    for f in ("PhysActivity", "Fruits", "Veggies"):
        if profile[f] == 0 and st.checkbox(S3.MUTABLE[f]):
            changes[f] = 1
    if not changes:
        st.info("Chọn ít nhất một thay đổi.")
        footer()
        return None
    try:
        out = S3.what_if(system, profile, changes, rules)
    except ValueError as e:
        st.error(str(e))
        footer()
        return None
    DB.whatif_view(profile, changes, out, system)
    footer()
    return out


def page_eval():
    DB.page_eval()
    footer()


def page_export(system, rules, profile):
    res = S3.assess(system, profile, rules)
    th = system["thresholds"]
    html(ui.result_card(res, th))
    html(ui.kpis([("Xác suất", f"{res['p']:.3f}", "đầu ra của mô hình"), ("Mức nguy cơ", res["tier_name"], f"ưu tiên: {res['priority_name']}"),
                  ("Quy tắc kích hoạt", str(len(res["rules"])), "có trong file xuất"), ("Hiệu chuẩn", U.CAL_LABEL[system["calibration"]], f"sơ đồ ngưỡng {th['scheme']}")]))
    st.subheader("Chọn định dạng tải về")
    c1, c2, c3 = st.columns(3)
    with c1:
        html('<div class="dss-card"><h4>Báo cáo HTML / PDF</h4><div class="dss-note">Một trang đẹp: đồng hồ, yếu tố ảnh hưởng, quy tắc có nguồn. Mở bằng trình duyệt rồi Ctrl+P để lưu PDF.</div></div>')
        st.download_button("Tải báo cáo (HTML)", ER.build_html(profile, res, system), "bao_cao_danh_gia.html", "text/html", type="primary")
    with c2:
        html('<div class="dss-card"><h4>Excel nhiều trang</h4><div class="dss-note">Tóm tắt, hồ sơ đầu vào, yếu tố ảnh hưởng, quy tắc. Đã định dạng màu sắc, sẵn sàng in.</div></div>')
        st.download_button("Tải Excel (.xlsx)", ER.build_excel(profile, res, system), "ket_qua_danh_gia.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    with c3:
        html('<div class="dss-card"><h4>CSV thô</h4><div class="dss-note">Bảng Mục – Nội dung đơn giản để đưa vào công cụ khác.</div></div>')
        df = S3.export_frame(profile, res, system)
        st.download_button("Tải CSV", df.to_csv(index=False).encode("utf-8-sig"), "ket_qua_danh_gia.csv", "text/csv")
    st.subheader("Xem trước nội dung file xuất")
    pv1, pv2, pv3, pv4 = st.tabs(["Tóm tắt", "Yếu tố ảnh hưởng", "Quy tắc", "Bảng CSV"])
    with pv1:
        st.dataframe(pd.DataFrame(ER.input_rows(profile), columns=["Thông tin", "Giá trị"]).astype(str), use_container_width=True, hide_index=True)
    with pv2:
        c = res["contributions"]
        st.dataframe(pd.DataFrame({"Yếu tố": c["Tên"], "Giá trị của hồ sơ": [ER.display_value(a, b) for a, b in zip(c["Đặc trưng"], c["Giá trị"])],
                                   "Đóng góp (log-odds)": c["SHAP (log-odds thô)"].round(3)}).astype(str), use_container_width=True, hide_index=True)
    with pv3:
        if res["rules"]:
            st.dataframe(pd.DataFrame([{"Mã": r["id"], "Tên quy tắc": r["name"], "Ưu tiên": S3.PRIORITY_NAMES[r["priority"]], "Hành động tham khảo": r["action"]}
                                       for r in sorted(res["rules"], key=lambda r: -r["priority"])]), use_container_width=True, hide_index=True)
        else:
            st.info("Không có quy tắc nào được kích hoạt.")
    with pv4:
        st.dataframe(df, use_container_width=True)
    footer()

# ----------------------------------------------------------------------------------------------
# Chức năng: chọn hồ sơ ưu tiên theo năng lực xử lý (Top-k)
# ----------------------------------------------------------------------------------------------
RANGES = {"BMI": (12, 98), "MentHlth": (0, 30), "PhysHlth": (0, 30), "GenHlth": (1, 5),
          "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)}      # các biến còn lại: 0 hoặc 1


def validate_batch(raw: pd.DataFrame):
    """Kiểm tra file nạp vào. Trả về (bảng sạch, danh sách lỗi)."""
    miss = [c for c in S3.FEATURES if c not in raw.columns]
    if miss:
        return None, [f"Thiếu cột: {', '.join(miss)}"]
    d = raw[S3.FEATURES].apply(pd.to_numeric, errors="coerce")
    errs = []
    bad = d.isna().any(axis=1)
    if bad.any():
        errs.append(f"{int(bad.sum())} dòng có giá trị thiếu hoặc không phải số (dòng đầu tiên: dòng {int(bad.idxmax()) + 2} trong file).")
    for c in S3.FEATURES:
        lo, hi = RANGES.get(c, (0, 1))
        out = d[c].notna() & ~d[c].between(lo, hi)
        if out.any():
            errs.append(f"Cột {c}: {int(out.sum())} dòng nằm ngoài khoảng {lo}–{hi}.")
        if c != "BMI":
            frac = d[c].notna() & (d[c] % 1 != 0)
            if frac.any():
                errs.append(f"Cột {c}: {int(frac.sum())} dòng không phải số nguyên.")
    return (None, errs) if errs else (d.reset_index(drop=True), [])


def demo_batch(system, n: int = 1000) -> pd.DataFrame:
    X = pd.DataFrame(system["X_ref"])[S3.FEATURES].astype(float)
    return X.sample(min(n, len(X)), random_state=0).reset_index(drop=True)


def page_capacity(system, rules):
    th = system["thresholds"]
    html('<div class="dss-card"><b>Chọn hồ sơ ưu tiên theo năng lực xử lý.</b> Khi đơn vị chỉ tiếp nhận được một số hồ sơ để tư vấn hoặc '
         'sàng lọc bổ sung, hệ thống xếp hạng các hồ sơ theo xác suất dự đoán và chọn k hồ sơ đầu danh sách (Top-k). '
         'Mức năng lực xử lý là <b>kịch bản giả định</b>, không phải nhu cầu sàng lọc đã được xác minh.</div>')
    src = st.radio("Nguồn dữ liệu", ["Dữ liệu minh họa (1.000 hồ sơ)", "Tải lên file CSV"], horizontal=True)
    if src.startswith("Dữ liệu"):
        df = demo_batch(system)
        st.caption("Dữ liệu minh họa lấy ngẫu nhiên từ tập huấn luyện, chỉ để thử chức năng, không dùng để đánh giá hiệu năng.")
    else:
        tmpl = pd.DataFrame([S3._neutral()])[S3.FEATURES].to_csv(index=False).encode("utf-8")
        st.download_button("Tải file CSV mẫu (21 cột)", tmpl, "mau_ho_so.csv", "text/csv")
        with st.expander("Ý nghĩa các mã (Age, GenHlth, Education, Income, Sex)"):
            for name, lab in (("Age", S3.AGE_LABELS), ("GenHlth", S3.GENHLTH_LABELS), ("Education", S3.EDUCATION_LABELS),
                              ("Income", S3.INCOME_LABELS), ("Sex", S3.SEX_LABELS)):
                st.markdown(f"**{name}**: " + "; ".join(f"{k} = {v}" for k, v in lab.items()))
            st.caption("Các biến có/không: 1 = Có, 0 = Không. BMI nhập số thực (12–98); MentHlth, PhysHlth là số ngày (0–30).")
        up = st.file_uploader("File CSV: mỗi dòng một hồ sơ, đủ 21 cột", type="csv")
        if up is None:
            st.info("Hãy tải file mẫu, điền các hồ sơ rồi tải lên; hoặc chuyển sang dữ liệu minh họa.")
            footer()
            return
        df, errs = validate_batch(pd.read_csv(up))
        if errs:
            for e in errs:
                st.error(e)
            footer()
            return
        if len(df) < 20:
            st.warning("Số hồ sơ quá ít, Top-k chỉ có ý nghĩa khi có nhiều hồ sơ.")

    n = len(df)
    k = st.select_slider("Năng lực xử lý: số hồ sơ có thể tiếp nhận trên 1.000 người", options=[100, 200, 300, 400], value=200)
    n_pick = max(1, round(k / 1000 * n))

    p = S3._prob(system, df)
    order = np.argsort(-p, kind="mergesort")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    tier = S3.assign_tier(p, th["t_low"], th["t_high"])
    mat = S3.rule_matrix(rules, df[S3.FEATURES], tier)
    p_k = float(p[order[n_pick - 1]])
    n_ge_med, n_high = int((tier != "low").sum()), int((tier == "high").sum())

    html(ui.kpis([("Số hồ sơ đã nạp", str(n), "một dòng một hồ sơ"),
                  ("Hồ sơ được ưu tiên (Top-k)", str(n_pick), f"{k} trên 1.000 người"),
                  ("Xác suất của hồ sơ cuối nhóm", f"{p_k:.3f}", "điểm cắt tương đương của Top-k"),
                  ("Ngưỡng cố định: Trung bình trở lên", str(n_ge_med), f"trong đó mức Cao: {n_high}")]))

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(p, bins=40, color="#19A7B8")
    ax.axvline(p_k, color="#B91C1C", lw=2, label=f"Điểm cắt Top-k = {p_k:.3f}")
    ax.axvline(th["t_low"], color="grey", ls=":", label=f"Ngưỡng Thấp/Trung bình = {th['t_low']:.3f}")
    ax.axvline(th["t_high"], color="grey", ls="--", label=f"Ngưỡng Trung bình/Cao = {th['t_high']:.3f}")
    ax.set_xlabel("Xác suất dự đoán"); ax.set_ylabel("Số hồ sơ"); ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("So sánh: ngưỡng cố định cho số hồ sơ cần xử lý thay đổi theo từng tập dữ liệu, còn Top-k luôn giữ đúng số hồ sơ đã chọn.")

    DB.capacity_tradeoff(k)

    res = pd.DataFrame({"Hạng": rank, "Xác suất": p.round(4), "Mức nguy cơ": [S3.TIER_NAMES[t] for t in tier],
                        "Ưu tiên Top-k": np.where(rank <= n_pick, "Có", "Không"), "Số quy tắc kích hoạt": mat.sum(axis=1).to_numpy(),
                        "Nhóm tuổi": df["Age"].map(S3.AGE_LABELS), "BMI": df["BMI"],
                        "Sức khỏe tự đánh giá": df["GenHlth"].map(S3.GENHLTH_LABELS), "Huyết áp cao": df["HighBP"].map(S3.YESNO)}).sort_values("Hạng")
    st.subheader("Danh sách hồ sơ đã xếp hạng")
    tab_top, tab_all = st.tabs([f"Top-k: {n_pick} hồ sơ ưu tiên", f"Toàn bộ {n} hồ sơ (có bộ lọc)"])
    with tab_top:
        st.dataframe(res.head(n_pick), use_container_width=True, hide_index=True)
    with tab_all:
        st.dataframe(DB.filtered(res, "cap_list", ["Mức nguy cơ", "Ưu tiên Top-k", "Nhóm tuổi", "Sức khỏe tự đánh giá", "Huyết áp cao"]),
                     use_container_width=True, hide_index=True)
    full = pd.concat([res[["Hạng", "Xác suất", "Mức nguy cơ", "Ưu tiên Top-k", "Số quy tắc kích hoạt"]].sort_index(), df], axis=1).sort_values("Hạng")
    st.download_button("Tải danh sách đã xếp hạng (CSV)", full.to_csv(index=False).encode("utf-8-sig"), "danh_sach_top_k.csv", "text/csv", type="primary")
    st.warning("Hồ sơ nằm ngoài Top-k không có nghĩa là không mắc bệnh. Top-k chỉ là cách ưu tiên khi nguồn lực có hạn; "
               "khuyến nghị tầm soát theo tuổi và yếu tố nguy cơ (R04, R05, R06, R12) vẫn áp dụng cho từng người.")
    footer()

def main():
    st.sidebar.markdown('<div class="dss-brand"><span style="font-size:1.8rem">🫀</span><div><b>DSS Tiểu đường</b><small>INS305 – Nhóm 01</small></div></div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Chức năng", PAGES, format_func=lambda k: f"{NAV_ICON[k]}  {k}")
    st.sidebar.markdown('<div class="dss-card" style="font-size:.85rem">Kết quả chỉ mang tính tham khảo từ dữ liệu khảo sát tự khai, không phải chẩn đoán y khoa.</div>', unsafe_allow_html=True)
    hero(page)
    if page in ("Dashboard", "Model Evaluation"):
        {"Dashboard": page_dashboard, "Model Evaluation": page_eval}[page]()
        return
    try:
        system, rules = get_system(), get_rules()
    except FileNotFoundError:
        st.error("Chưa có models/Phase3_system.joblib hoặc rules/decision_rules.json – hãy chạy xong notebook Giai đoạn 3 (mục 3.5 và 3.6).")
        return
    except ModuleNotFoundError as e:
        st.error(f"Python đang chạy ứng dụng chưa có thư viện '{e.name}' (cần để nạp mô hình).")
        st.code(f'"{sys.executable}" -m pip install {e.name}\n\n# hoặc chạy ứng dụng bằng đúng Python đã dùng cho notebook:\n"<đường dẫn python của notebook>" -m streamlit run app.py', language="bash")
        return
    if page == "Capacity Planning":
        page_capacity(system, rules)    
        return
    profile = profile_form()
    {"Risk Assessment": page_risk, "Explainable AI": page_xai, "Decision Support": page_decision, "What-if Analysis": page_whatif,
     "Export Results": page_export}[page](system, rules, profile)


if __name__ == "__main__":
    main()
