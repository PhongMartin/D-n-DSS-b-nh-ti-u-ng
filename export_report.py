"""Xuất kết quả đánh giá: (1) sổ Excel nhiều trang đã định dạng, (2) báo cáo HTML một trang (mở trong trình duyệt, in ra PDF được)."""
from __future__ import annotations

import io
from datetime import datetime
from html import escape

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import dss_stage3 as S3
import dss_utils as U
import ui

INK, TEAL, SOFT, LINE = "0A2A33", "0E5A66", "E8F1F3", "D5E3E6"
TIER_FILL = {"low": ("DCFCE7", "14532D"), "medium": ("FEF3C7", "78350F"), "high": ("FEE2E2", "7F1D1D")}
TIER_ICON = {"low": "▼", "medium": "◆", "high": "▲"}
thin = Side(style="thin", color=LINE)
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)


def display_value(k, v):
    if k in S3.VALUE_LABELS:
        return S3.VALUE_LABELS[k].get(v, v)
    if k in S3.BINARY_FEATURES:
        return S3.YESNO.get(v, v)
    return v


def input_rows(profile: dict) -> list[tuple[str, object]]:
    return [(S3.FEATURE_VN[k], display_value(k, v)) for k, v in profile.items()]


def _title(ws, text, sub, width):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=width)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=width)
    ws["A1"], ws["A2"] = text, sub
    ws["A1"].font = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    ws["A2"].font = Font(name="Calibri", size=10, color="BFE3E8")
    for r in (1, 2):
        for c in range(1, width + 1):
            ws.cell(r, c).fill = PatternFill("solid", fgColor=INK)
    ws.row_dimensions[1].height = 30
    ws.row_dimensions[2].height = 18
    ws["A1"].alignment = Alignment(vertical="center", indent=1)
    ws["A2"].alignment = Alignment(vertical="center", indent=1)


def _header(ws, row, labels):
    for i, t in enumerate(labels, 1):
        c = ws.cell(row, i, t)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=TEAL)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BOX
    ws.row_dimensions[row].height = 26


def _body(cell, zebra=False, wrap=True, align="left"):
    cell.border = BOX
    cell.alignment = Alignment(vertical="top", wrap_text=wrap, horizontal=align)
    if zebra:
        cell.fill = PatternFill("solid", fgColor="F5FAFB")


def _widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _print(ws):
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.sheet_view.showGridLines = False


def build_excel(profile: dict, res: dict, system: dict) -> bytes:
    th = system["thresholds"]
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    sub = f"INS305 – Nhóm 01 · Xuất lúc {stamp} · Chỉ mang tính tham khảo, không phải chẩn đoán"
    wb = Workbook()

    # 1. Tóm tắt
    ws = wb.active
    ws.title = "Tóm tắt"
    _title(ws, "Kết quả đánh giá nguy cơ tiền tiểu đường / tiểu đường", sub, 3)
    fill, ink = TIER_FILL[res["tier"]]
    rows = [("Xác suất dự đoán", round(res["p"], 3), system["target_meaning"]),
            ("Mức nguy cơ", f"{TIER_ICON[res['tier']]} {res['tier_name']}", f"Thấp < {th['t_low']:.3f} ≤ Trung bình < {th['t_high']:.3f} ≤ Cao (sơ đồ {th['scheme']})"),
            ("Mức ưu tiên hỗ trợ", res["priority_name"], "Mức ưu tiên cao nhất trong các quy tắc được kích hoạt và mức nguy cơ"),
            ("Số quy tắc được kích hoạt", len(res["rules"]), "Chi tiết ở trang 'Quy tắc'"),
            ("Mô hình", f"{system['model_name']} (seed {system['seed']})", f"Hiệu chuẩn xác suất: {U.CAL_LABEL[system['calibration']]}")]
    _header(ws, 4, ["Chỉ số", "Giá trị", "Ghi chú"])
    for i, (a, b, c) in enumerate(rows, 5):
        for j, v in enumerate((a, b, c), 1):
            cell = ws.cell(i, j, v)
            _body(cell, zebra=i % 2 == 0)
        ws.cell(i, 1).font = Font(bold=True, color=INK)
        ws.cell(i, 2).font = Font(bold=True, size=12, color=INK)
        ws.row_dimensions[i].height = 32
    ws["B5"].number_format = "0.000"
    ws["B6"].fill = PatternFill("solid", fgColor=fill)
    ws["B6"].font = Font(bold=True, size=12, color=ink)
    ws.merge_cells("A11:C11")
    ws["A11"] = "Lưu ý: " + res["disclaimer"]
    ws["A11"].alignment = Alignment(wrap_text=True, vertical="top")
    ws["A11"].fill = PatternFill("solid", fgColor=SOFT)
    ws["A11"].font = Font(italic=True, color="52666B")
    ws.row_dimensions[11].height = 48
    _widths(ws, [28, 26, 70])
    _print(ws)

    # 2. Hồ sơ đầu vào
    ws = wb.create_sheet("Hồ sơ đầu vào")
    _title(ws, "Thông tin hồ sơ đã nhập", sub, 2)
    _header(ws, 4, ["Thông tin", "Giá trị"])
    for i, (k, v) in enumerate(input_rows(profile), 5):
        _body(ws.cell(i, 1, k), i % 2 == 0)
        _body(ws.cell(i, 2, v), i % 2 == 0, align="center")
    _widths(ws, [42, 28])
    ws.freeze_panes = "A5"
    _print(ws)

    # 3. Yếu tố SHAP
    ws = wb.create_sheet("Yếu tố ảnh hưởng")
    _title(ws, "Yếu tố ảnh hưởng đến kết quả của hồ sơ này (SHAP)", sub, 5)
    c = res["contributions"]
    _header(ws, 4, ["Yếu tố", "Giá trị của hồ sơ", "Đóng góp (log-odds)", "Chiều ảnh hưởng", "Mức độ"])
    m = float(c["SHAP (log-odds thô)"].abs().max()) or 1.0
    for i, (_, r) in enumerate(c.iterrows(), 5):
        v = float(r["SHAP (log-odds thô)"])
        up = v > 0
        vals = [r["Tên"], display_value(r["Đặc trưng"], r["Giá trị"]), round(v, 3), "▲ làm tăng nguy cơ" if up else "▼ làm giảm nguy cơ", "█" * max(1, round(10 * abs(v) / m))]
        for j, x in enumerate(vals, 1):
            cell = ws.cell(i, j, x)
            _body(cell, i % 2 == 0, align="center" if j > 1 else "left")
        ws.cell(i, 3).number_format = "+0.000;-0.000"
        col = "B91C1C" if up else "0369A1"
        for j in (3, 4, 5):
            ws.cell(i, j).font = Font(bold=(j != 5), color=col)
    n = 5 + len(c)
    ws.merge_cells(start_row=n + 1, start_column=1, end_row=n + 1, end_column=5)
    ws.cell(n + 1, 1, f"Thang đo: log-odds đầu ra thô của LightGBM. Nền = {res['shap_base_logodds']:.2f}; tổng = {res['shap_total_logodds']:.2f}. "
                      "SHAP mô tả cách mô hình dùng biến, không phải quan hệ nhân quả.").alignment = Alignment(wrap_text=True, vertical="top")
    ws.cell(n + 1, 1).font = Font(italic=True, color="52666B")
    ws.row_dimensions[n + 1].height = 34
    _widths(ws, [34, 22, 20, 26, 16])
    _print(ws)

    # 4. Quy tắc
    ws = wb.create_sheet("Quy tắc")
    _title(ws, "Quy tắc được kích hoạt và hành động tham khảo", sub, 8)
    _header(ws, 4, ["Mã", "Tên quy tắc", "Loại", "Ưu tiên", "Hành động tham khảo", "Nguồn", "Tài liệu / mục", "Trạng thái xác minh"])
    pf = {1: "DBEAFE", 2: "FEF3C7", 3: "FEE2E2"}
    for i, r in enumerate(sorted(res["rules"], key=lambda r: -r["priority"]), 5):
        src = r["source"] + (f" — {r['verification']}" if r.get("verification") else "")
        vals = [r["id"], r["name"], "Chuyên môn" if r["type"] == "guideline" else "Vận hành (nhóm xây dựng)", S3.PRIORITY_NAMES[r["priority"]], r["action"], src,
                (r.get("doc", "") + " – " + r.get("section", "")).strip(" –"), r.get("status", "")]
        for j, x in enumerate(vals, 1):
            _body(ws.cell(i, j, x), i % 2 == 0, align="center" if j in (1, 4) else "left")
        ws.cell(i, 4).fill = PatternFill("solid", fgColor=pf[r["priority"]])
        ws.cell(i, 4).font = Font(bold=True, color=INK)
        ws.row_dimensions[i].height = 62
    if not res["rules"]:
        ws.cell(5, 1, "Không có quy tắc nào được kích hoạt.")
    _widths(ws, [8, 30, 22, 14, 60, 52, 44, 40])
    ws.freeze_panes = "A5"
    _print(ws)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_html(profile: dict, res: dict, system: dict) -> bytes:
    """Báo cáo một trang, tự chứa (không cần mạng ngoài font), in ra PDF bằng Ctrl+P."""
    th = system["thresholds"]
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    inputs = "".join(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in input_rows(profile))
    c = res["contributions"]
    factors = ui.factor_bars(list(zip(c["Tên"], c["SHAP (log-odds thô)"])), top=8)
    rules = "".join(ui.rule_card(r, S3.PRIORITY_NAMES) for r in sorted(res["rules"], key=lambda r: -r["priority"])) or "<p>Không có quy tắc nào được kích hoạt.</p>"
    css = """
    :root{--ink:#0A2A33;--teal:#0E5A66;--aqua:#19A7B8;--line:#D5E3E6;--muted:#52666B}
    *{box-sizing:border-box} body{font-family:'Be Vietnam Pro','Segoe UI',sans-serif;color:var(--ink);background:#F3F7F8;margin:0;line-height:1.55}
    .page{max-width:900px;margin:0 auto;padding:24px}
    h1,h2,h3,h4{font-family:'Bricolage Grotesque','Segoe UI',sans-serif;letter-spacing:-.01em}
    .dss-hero{background:linear-gradient(120deg,#0A2A33,#0E5A66);border-radius:18px;padding:26px 32px;margin-bottom:18px;color:#fff}
    .dss-hero h1{color:#fff;margin:4px 0;font-size:1.8rem}.dss-hero p{color:#BFE3E8;margin:0}
    .dss-eyebrow{color:#7FD3DD;font-size:.75rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase}
    .dss-card,.dss-result{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 22px;margin-bottom:14px}
    .dss-result{display:flex;flex-wrap:wrap;gap:8px 28px;align-items:center}.dss-result .gauge{flex:0 0 280px}.dss-result .info{flex:1 1 240px}
    .dss-big{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:2.2rem;line-height:1}
    .dss-badge{display:inline-block;font-weight:600;padding:5px 14px;border-radius:999px;border:2px solid;margin-bottom:8px}
    .dss-low{color:#14532D;background:#DCFCE7;border-color:#14532D}.dss-medium{color:#78350F;background:#FEF3C7;border-color:#78350F}.dss-high{color:#7F1D1D;background:#FEE2E2;border-color:#7F1D1D}
    .dss-note{color:var(--muted);font-size:.88rem}
    table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}
    td,th{padding:7px 14px;border-bottom:1px solid var(--line);text-align:left;font-size:.92rem} th{background:var(--teal);color:#fff}
    tr:nth-child(even) td{background:#F5FAFB}
    @media print{body{background:#fff}.page{padding:0}.dss-hero,.dss-badge,.dss-factor .bar,.dss-rule{-webkit-print-color-adjust:exact;print-color-adjust:exact}.dss-rule,.dss-card,.dss-result{break-inside:avoid}}
    """
    body = (f'<div class="page"><div class="dss-hero"><div class="dss-eyebrow">INS305 · Hệ hỗ trợ quyết định · Nhóm 01</div><h1>Kết quả đánh giá nguy cơ tiền tiểu đường / tiểu đường</h1>'
            f'<p>Xuất lúc {stamp} · {escape(system["model_name"])} · hiệu chuẩn: {escape(U.CAL_LABEL[system["calibration"]])}</p></div>'
            f'{ui.result_card(res, th)}'
            f'<p class="dss-note">Ngưỡng phân mức: Thấp &lt; {th["t_low"]:.3f} ≤ Trung bình &lt; {th["t_high"]:.3f} ≤ Cao (sơ đồ {escape(str(th["scheme"]))}).</p>'
            f'<h2>Yếu tố ảnh hưởng</h2>{factors}<h2>Quy tắc và hành động tham khảo</h2>{rules}'
            f'<h2>Hồ sơ đã nhập</h2><table><tr><th>Thông tin</th><th>Giá trị</th></tr>{inputs}</table>'
            f'{ui.footer(res["disclaimer"] + " Kết quả dựa trên dữ liệu khảo sát tự khai (BRFSS 2015), không phải chẩn đoán y khoa.")}</div>')
    page = f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kết quả đánh giá nguy cơ</title>' \
           f"<link href=\"https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=Be+Vietnam+Pro:wght@400;500;600&display=swap\" rel=\"stylesheet\"><style>{css}</style>{ui.EXTRA_CSS}</head><body>{body}</body></html>"
    return page.encode("utf-8")
