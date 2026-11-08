"""Thành phần giao diện (HTML/SVG) cho app.py – dùng bảng màu và font của bản app_test (Bricolage Grotesque + Be Vietnam Pro).
Mức nguy cơ luôn có biểu tượng + chữ, không chỉ dựa vào màu."""
from __future__ import annotations

import math
from html import escape

EXTRA_CSS = """
<style>
.dss-eyebrow { position:relative; color:#7FD3DD; font-size:.78rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase; margin-bottom:8px; }
.dss-kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-bottom:18px; }
.dss-kpi { background:#fff; border:1px solid var(--line); border-top:4px solid var(--aqua); border-radius:14px; padding:14px 18px; box-shadow:0 6px 20px rgba(10,42,51,.06); }
.dss-kpi .l { color:var(--muted); font-size:.82rem; }
.dss-kpi .v { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.75rem; line-height:1.15; }
.dss-kpi .s { color:var(--muted); font-size:.8rem; }
.dss-steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-bottom:18px; }
.dss-step { position:relative; background:#fff; border:1px solid var(--line); border-radius:16px; padding:16px 18px 14px; }
.dss-step .n { display:inline-flex; width:30px; height:30px; border-radius:50%; background:var(--ink); color:#fff; font-weight:700; align-items:center; justify-content:center; margin-bottom:8px; }
.dss-step b { display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:1.05rem; }
.dss-step span { color:var(--muted); font-size:.88rem; }
.dss-factor { display:grid; grid-template-columns:minmax(120px,210px) 1fr auto; gap:10px; align-items:center; margin:9px 0; }
.dss-factor .track { height:10px; background:#EAF1F3; border-radius:6px; overflow:hidden; }
.dss-factor .bar { height:100%; border-radius:6px; }
.dss-factor .val { font-weight:600; white-space:nowrap; }
.dss-rule { background:#fff; border:1px solid var(--line); border-left:8px solid; border-radius:14px; padding:16px 20px; margin-bottom:12px; box-shadow:0 6px 20px rgba(10,42,51,.05); }
.dss-rule.p1 { border-left-color:#0369A1; } .dss-rule.p2 { border-left-color:#B45309; } .dss-rule.p3 { border-left-color:#B91C1C; }
.dss-rule .t { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.08rem; }
.dss-tag { display:inline-block; font-size:.78rem; font-weight:600; padding:2px 10px; border-radius:999px; margin:6px 6px 0 0; border:1px solid; }
.dss-tag.g { color:#065F46; background:#D1FAE5; border-color:#065F46; } .dss-tag.o { color:#3730A3; background:#E0E7FF; border-color:#3730A3; }
.dss-tag.pr { color:#0A2A33; background:#E8F1F3; border-color:#0A2A33; }
.dss-rule .a { margin:10px 0 4px; } .dss-rule .e { color:var(--muted); font-size:.92rem; } .dss-rule .s { color:#6B7F84; font-size:.8rem; margin-top:8px; }
.dss-compare { display:grid; grid-template-columns:1fr auto 1fr; gap:14px; align-items:center; }
.dss-compare .box { background:#fff; border:1px solid var(--line); border-radius:20px; padding:14px 18px 18px; text-align:center; }
.dss-compare .box h4 { margin:0; } .dss-compare .box svg { width:100%; max-width:280px; }
.dss-delta { text-align:center; font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.9rem; line-height:1.1; }
.dss-foot { margin-top:26px; padding:12px 18px; border-radius:12px; background:#E8F1F3; color:#52666B; font-size:.85rem; }
.dss-hero svg { opacity:.35 !important; }
.dss-result .gauge { flex:0 0 310px !important; }
.dss-legend { display:flex; gap:16px; flex-wrap:wrap; font-size:.85rem; color:var(--muted); margin-top:4px; }
@media (max-width: 760px) { .dss-compare { grid-template-columns:1fr; } }
</style>
"""

TIER_ICON = {"low": "▼ ", "medium": "◆ ", "high": "▲ "}
TIER_COL = {"low": "#34A36B", "medium": "#E9A93B", "high": "#D64545"}


def kpis(items: list[tuple[str, str, str]]) -> str:
    body = "".join(f'<div class="dss-kpi"><div class="l">{escape(l)}</div><div class="v">{escape(v)}</div><div class="s">{escape(s)}</div></div>' for l, v, s in items)
    return f'<div class="dss-kpis">{body}</div>'


def steps(items: list[tuple[str, str]]) -> str:
    body = "".join(f'<div class="dss-step"><div class="n">{i}</div><b>{escape(t)}</b><span>{escape(d)}</span></div>' for i, (t, d) in enumerate(items, 1))
    return f'<div class="dss-steps">{body}</div>'


def badge(tier: str, tier_name: str) -> str:
    return f'<span class="dss-badge dss-{tier}" role="status">{TIER_ICON[tier]}Mức nguy cơ: {escape(tier_name)}</span>'


def _pt(f: float, r: float):
    a = math.pi * (1 - f)
    return 150 + r * math.cos(a), 150 - r * math.sin(a)


def gauge_svg(p: float, tl: float, th: float, animate: bool = True) -> str:
    """Đồng hồ 3 vùng chia đều trên màn hình (Thấp/Trung bình/Cao); kim nằm đúng vùng theo ngưỡng; có nhãn vùng và giá trị ngưỡng."""
    f = (p / tl) / 3 if p < tl else (1 / 3 + (p - tl) / (th - tl) / 3 if p < th else 2 / 3 + (p - th) / max(1 - th, 1e-9) / 3)
    f = min(max(f, 0.02), 0.98)
    arcs, labs = "", ""
    zones = ((0, 1 / 3, "#34A36B", "Thấp"), (1 / 3, 2 / 3, "#E9A93B", "Trung bình"), (2 / 3, 1, "#D64545", "Cao"))
    for a, b, col, name in zones:
        (x0, y0), (x1, y1) = _pt(a + 0.006, 112), _pt(b - 0.006, 112)
        arcs += f'<path d="M{x0:.1f} {y0:.1f} A112 112 0 0 1 {x1:.1f} {y1:.1f}" fill="none" stroke="{col}" stroke-width="20" stroke-linecap="round"/>'
        lx, ly = _pt((a + b) / 2, 138)
        labs += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#0A2A33">{name}</text>'
    for fr, v in ((1 / 3, tl), (2 / 3, th)):
        tx, ty = _pt(fr, 131)
        labs += f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="10" font-weight="500" fill="#52666B">{v:.3f}</text>'
    nx, ny = _pt(f, 92)
    cls = ' class="needle"' if animate else ""
    return (f'<svg viewBox="0 0 300 192" role="img" aria-label="Đồng hồ mức nguy cơ, xác suất {p:.3f}">{arcs}{labs}'
            f'<g{cls}><line x1="150" y1="150" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#0A2A33" stroke-width="5" stroke-linecap="round"/></g>'
            f'<circle cx="150" cy="150" r="11" fill="#0A2A33"/><text x="150" y="184" text-anchor="middle" font-size="16" font-weight="700" fill="#0A2A33">{p:.3f}</text></svg>')


def result_card(res: dict, th: dict | None) -> str:
    g = f'<div class="gauge">{gauge_svg(res["p"], th["t_low"], th["t_high"])}</div>' if th else ""
    return (f'<div class="dss-result">{g}<div class="info">{badge(res["tier"], res["tier_name"])}<div class="dss-big">{res["p"]:.3f}</div>'
            f'<div>Mức ưu tiên hỗ trợ: <b>{escape(res["priority_name"])}</b></div><div class="dss-note">{escape(res["model_note"])}</div></div></div>')


def factor_bars(rows: list[tuple[str, float]], top: int = 5) -> str:
    rows = rows[:top]
    m = max((abs(v) for _, v in rows), default=1) or 1
    out = ""
    for name, v in rows:
        col, arrow, word = ("#B91C1C", "▲", "làm tăng") if v > 0 else ("#0369A1", "▼", "làm giảm")
        out += (f'<div class="dss-factor"><span>{escape(str(name))}</span><span class="track"><span class="bar" style="display:block;width:{100 * abs(v) / m:.0f}%;background:{col}"></span></span>'
                f'<span class="val" style="color:{col}">{arrow} {word} ({v:+.2f})</span></div>')
    return (f'<div class="dss-card"><h4 style="margin:0 0 4px">Yếu tố ảnh hưởng chính của hồ sơ này</h4>{out}'
            '<div class="dss-note">Đơn vị: log-odds của mô hình thô (SHAP). Mô tả cách mô hình dùng biến, không phải quan hệ nhân quả.</div></div>')


def rule_card(r: dict, prio_names: dict) -> str:
    tag = '<span class="dss-tag g">Chuyên môn</span>' if r["type"] == "guideline" else '<span class="dss-tag o">Vận hành (nhóm xây dựng)</span>'
    ver = f' — {escape(r["verification"])}' if r.get("verification") else ""
    return (f'<div class="dss-rule p{r["priority"]}"><div class="t">{escape(r["id"])} · {escape(r["name"])}</div>'
            f'{tag}<span class="dss-tag pr">Ưu tiên: {escape(prio_names[r["priority"]])}</span>'
            f'<div class="a"><b>Hành động tham khảo:</b> {escape(r["action"])}</div><div class="e">{escape(r["explanation"])}</div>'
            f'<div class="s">Nguồn: {escape(r["source"])}{ver}</div>'
            + (f'<div class="s">Tài liệu/mục: {escape(r["doc"])} – {escape(r["section"])}'
               + (f' (khuyến cáo {escape(r["rec_no"])})' if r.get("rec_no") else '') + '</div>' if r.get("doc") else '')
            + (f'<div class="s">Trạng thái xác minh: {escape(r["status"])}</div>' if r.get("status") else '')
            + (f'<div class="s">Ranh giới: {escape(r["boundary_note"])}</div>' if r.get("boundary_note") else '') + '</div>')


def compare(before: dict, after: dict, delta: float, th: dict) -> str:
    def box(title, res):
        return (f'<div class="box"><h4>{title}</h4>{gauge_svg(res["p"], th["t_low"], th["t_high"], animate=False)}<div>{badge(res["tier"], res["tier_name"])}</div></div>')
    col = "#0369A1" if delta < 0 else "#B91C1C"
    mid = f'<div><div class="dss-delta" style="color:{col}">{"▼" if delta < 0 else "▲"} {delta:+.3f}</div><div class="dss-note" style="text-align:center">thay đổi xác suất</div></div>'
    return f'<div class="dss-compare">{box("Trước", before)}{mid}{box("Sau", after)}</div>'


def footer(text: str) -> str:
    return f'<div class="dss-foot">{escape(text)}</div>'
