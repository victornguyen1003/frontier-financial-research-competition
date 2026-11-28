"""Interactive Altair charts for the viewer's Data analysis page. Pure functions of the analysis
tables (no Streamlit), so they can be tested. Persona labels run down the y-axis so that any
number of personas stays readable; the page passes in only the personas the reader picked."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from llm_investors.analysis.style import ARCH_COLORS, GRID, INK_2, SERIES

ARCHS = ["none", "log", "reflect"]
POOLED = "profiles (pooled)"
CONTROLS = ["C0", "C1", POOLED]
ARM_COLORS = {**ARCH_COLORS, "all arms": INK_2}
REF_COLORS = {"C0": SERIES[1], "C1": "#e34948", POOLED: "#0b0b0b"}


def arm_scale(arms) -> alt.Scale:
    """Arm colors, with only the arms in use in the legend."""
    dom = [a for a in ARCHS + ["all arms"] if a in set(arms)]
    return alt.Scale(domain=dom, range=[ARM_COLORS[a] for a in dom])
RATIO_FLOOR = 0.05   # lower CI ends at or below 0 are drawn here on the log axis
METRICS = {"DE": "DE = PGR − PLR", "PGR": "PGR", "PLR": "PLR", "Ratio": "PGR / PLR (log scale)"}


def persona_order(personas) -> list[str]:
    s = set(personas)
    return [p for p in CONTROLS if p in s] + sorted(p for p in s if p not in CONTROLS)


def rank_by(cells: pd.DataFrame, metric: str = "DE") -> list[str]:
    """Personas ordered by `metric` pooled over all arms, highest first."""
    a = cells[cells["architecture"] == "all arms"].set_index("persona")[metric]
    return list(a.sort_values(ascending=False, na_position="last").index)


def _fmt(cols):
    return [alt.Tooltip(c, format=".3f") if c not in ("persona", "architecture", "runs")
            else c for c in cols]


def cells_chart(cells: pd.DataFrame, personas: list[str], arms: list[str], metric: str = "DE",
                benchmarks: pd.DataFrame | None = None) -> tuple[alt.Chart | None, list[str]]:
    """Horizontal bars (or points for Ratio) per persona, one per memory arm, with 95% CIs.
    Returns the chart and the personas dropped because the metric is undefined (Ratio with
    PLR = 0)."""
    d = cells[cells["persona"].isin(personas) & cells["architecture"].isin(arms)].copy()
    lo, hi = f"{metric}_lo", f"{metric}_hi"
    bad = d[~np.isfinite(d[metric].astype(float))]
    dropped = sorted(set(bad["persona"] + " (" + bad["architecture"] + ")"))
    d = d[np.isfinite(d[metric].astype(float))]
    if d.empty:
        return None, dropped
    log = metric == "Ratio"
    if log:
        d[lo] = d[lo].clip(lower=RATIO_FLOOR)
        d = d[d[metric] > 0]
    order = [p for p in personas if p in set(d["persona"])]
    n_arms = d["architecture"].nunique()
    y = alt.Y("persona:N", sort=order, title=None,
              axis=alt.Axis(labelFontSize=12, labelLimit=160))
    xs = alt.Scale(type="log") if log else alt.Scale(zero=not log)
    x = alt.X(f"{metric}:Q", title=METRICS[metric], scale=xs)
    off = alt.YOffset("architecture:N", sort=[a for a in ARCHS + ["all arms"] if a in arms])
    color = alt.Color("architecture:N", scale=arm_scale(d["architecture"]), title="memory arm",
                      legend=alt.Legend(orient="top"))
    tip = _fmt(["persona", "architecture", "runs", "PGR", "PLR", "DE", "DE_lo", "DE_hi",
                "Ratio"])
    base = alt.Chart(d).encode(y=y, yOffset=off, color=color, tooltip=tip)
    mark = base.mark_point(filled=True, size=70) if log else base.mark_bar(height={"band": 0.9})
    err = base.mark_rule(color=INK_2, strokeWidth=1.2).encode(x=f"{lo}:Q", x2=f"{hi}:Q")
    layers = [mark.encode(x=x), err]
    ref = 1.0 if log else 0.0
    layers.append(alt.Chart(pd.DataFrame({"v": [ref]})).mark_rule(color=INK_2)
                  .encode(x="v:Q"))
    if log and benchmarks is not None and not benchmarks.empty:
        bm = benchmarks.rename(columns={"ratio": "v"})
        bm = bm.assign(label=bm["investor_type"] + " " + bm["v"].map("{:.2f}".format))
        layers.append(alt.Chart(bm).mark_rule(strokeDash=[5, 4], color=SERIES[1])
                      .encode(x="v:Q", tooltip=["investor_type", alt.Tooltip("v:Q",
                                                                             format=".2f")]))
        layers.append(alt.Chart(bm).mark_text(angle=270, align="left", dx=4, dy=-4,
                                              color=SERIES[1], fontSize=11)
                      .encode(x="v:Q", y=alt.value(0),
                              text=alt.Text("label:N")))
    h = max(160, len(order) * (14 * n_arms + 10))
    return alt.layer(*layers).properties(height=h), dropped


def memory_chart(cells: pd.DataFrame, personas: list[str], change: bool = False,
                 background: bool = False) -> alt.Chart | None:
    """DE across the memory arms, one line per persona. `change` plots DE(arm) − DE(none).
    `background` draws every other profile as faint grey lines."""
    c = cells[cells["architecture"].isin(ARCHS)][["persona", "architecture", "DE", "runs"]]
    if change:
        none = c[c["architecture"] == "none"].set_index("persona")["DE"]
        c = c.assign(DE=c["DE"] - c["persona"].map(none))
    ytitle = "DE(arm) − DE(none)" if change else "DE = PGR − PLR"
    x = alt.X("architecture:N", sort=ARCHS, title="memory arm",
              axis=alt.Axis(labelAngle=0, labelFontSize=12))
    y = alt.Y("DE:Q", title=ytitle)
    sel = c[c["persona"].isin(personas)]
    if sel.empty:
        return None
    hover = alt.selection_point(fields=["persona"], on="pointerover", clear="pointerout")
    present = list(dict.fromkeys(p for p in personas if p in set(sel["persona"])))
    rng = [REF_COLORS.get(p, SERIES[0]) for p in present]
    color = alt.Color("persona:N", scale=alt.Scale(domain=present, range=rng), legend=None)
    enc = dict(x=x, y=y, color=color, detail="persona:N")
    lines = (alt.Chart(sel).mark_line(point=True, strokeWidth=2.5)
             .encode(**enc, opacity=alt.condition(hover, alt.value(1), alt.value(0.3)),
                     tooltip=["persona", "architecture", "runs",
                              alt.Tooltip("DE:Q", format=".3f", title=ytitle)])
             .add_params(hover))
    labels = (alt.Chart(sel[sel["architecture"] == ARCHS[-1]])
              .mark_text(align="left", dx=8, fontSize=11)
              .encode(**enc, text="persona:N",
                      opacity=alt.condition(hover, alt.value(1), alt.value(0.5))))
    layers = []
    if background:
        rest = c[~c["persona"].isin(personas) & ~c["persona"].isin(CONTROLS)]
        layers.append(alt.Chart(rest).mark_line(color=GRID, strokeWidth=1)
                      .encode(x=x, y=y, detail="persona:N", tooltip=["persona"]))
    layers += [alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=INK_2).encode(y="v:Q"),
               lines, labels]
    return alt.layer(*layers).properties(height=380)


def forest_chart(blind: pd.DataFrame, band: float, order: list[str]) -> alt.Chart:
    """Attribute effects d_k with 95% CIs, the blind-spot band and the mean Holm MDE."""
    b = blind.assign(result=np.where(blind["significant_holm"].astype(bool),
                                     "significant (Holm)", "not significant"))
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12,
                                                               labelLimit=200))
    color = alt.Color("result:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["significant (Holm)", "not significant"],
                                      range=[SERIES[0], INK_2]))
    tip = ["label", alt.Tooltip("coef:Q", format="+.3f", title="d_k"),
           alt.Tooltip("ci_lo:Q", format="+.3f", title="CI low"),
           alt.Tooltip("ci_hi:Q", format="+.3f", title="CI high"),
           alt.Tooltip("p:Q", format=".4f"), alt.Tooltip("p_holm:Q", format=".4f"),
           alt.Tooltip("mde_holm:Q", format=".3f", title="MDE (Holm)")]
    base = alt.Chart(b).encode(y=y, color=color, tooltip=tip)
    m = float(b["mde_holm"].mean())
    bandc = alt.Chart(pd.DataFrame({"a": [-band], "b": [band]})).mark_rect(
        color=GRID, opacity=0.8).encode(x="a:Q", x2="b:Q")
    mdes = alt.Chart(pd.DataFrame({"v": [-m, m]})).mark_rule(
        strokeDash=[2, 3], color=SERIES[1]).encode(x="v:Q")
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=INK_2).encode(x="v:Q")
    lo = min(float(b["ci_lo"].min()), -m, -band)
    hi = max(float(b["ci_hi"].max()), m, band)
    pad = 0.05 * (hi - lo)
    pts = base.mark_point(filled=True, size=80).encode(
        x=alt.X("coef:Q", title="d_k: change in P(sell | gain) − P(sell | loss)",
                scale=alt.Scale(domain=[lo - pad, hi + pad], nice=False)))
    err = base.mark_rule(strokeWidth=2).encode(x="ci_lo:Q", x2="ci_hi:Q")
    return alt.layer(bandc, mdes, zero, err, pts).properties(height=max(160, 34 * len(b)))


def path_frames(events_by_arm: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices and trades of one path for several arms, from run logs."""
    first = next(iter(events_by_arm.values()))
    start = next(e for e in first if e["type"] == "run_start")
    prices = np.array(start["prices"])
    px = pd.DataFrame([{"stock": s, "month": t, "price": float(prices[t, j])}
                       for j, s in enumerate(start["labels"]) for t in range(len(prices))])
    rows = []
    for i, (arm, ev) in enumerate(events_by_arm.items()):
        why = {e["period"]: e.get("reasoning", "") for e in ev if e["type"] == "period_end"}
        for e in ev:
            if e["type"] == "trade":
                r = why.get(e["period"], "") or ""
                rows.append({"arm": arm, "stock": e["stock"], "month": e["period"],
                             "action": e["action"], "quantity": e["quantity"],
                             "price": e["price"], "avg_cost": e.get("avg_cost"),
                             "y": e["price"] * (1 + 0.03 * (i - 1)),
                             "reasoning": r if len(r) <= 300 else r[:297] + "..."})
    tr = pd.DataFrame(rows, columns=["arm", "stock", "month", "action", "quantity", "price",
                                     "avg_cost", "y", "reasoning"])
    return px, tr


def path_chart(px: pd.DataFrame, tr: pd.DataFrame, stocks: list[str]) -> alt.Chart:
    """One panel per stock: the price line and each arm's trades (▼ sell, ▲ buy)."""
    panels = []
    for s in stocks:
        p = px[px["stock"] == s]
        line = alt.Chart(p).mark_line(color=INK_2).encode(
            x=alt.X("month:Q", title="month", scale=alt.Scale(domain=[0, int(px["month"].max())])),
            y=alt.Y("price:Q", title="price", scale=alt.Scale(zero=False)),
            tooltip=["month", alt.Tooltip("price:Q", format=".2f")])
        start = alt.Chart(pd.DataFrame({"v": [100.0]})).mark_rule(color=GRID).encode(y="v:Q")
        t = tr[tr["stock"] == s]
        pts = alt.Chart(t).mark_point(filled=True, size=110).encode(
            x="month:Q", y="y:Q",
            color=alt.Color("arm:N", scale=arm_scale(tr["arm"]), title="memory arm",
                            legend=alt.Legend(orient="top")),
            shape=alt.Shape("action:N", scale=alt.Scale(domain=["sell", "buy"],
                                                        range=["triangle-down",
                                                               "triangle-up"]),
                            legend=alt.Legend(orient="top")),
            tooltip=["arm", "action", "month", "quantity",
                     alt.Tooltip("price:Q", format=".2f"),
                     alt.Tooltip("avg_cost:Q", format=".2f", title="avg cost"),
                     "reasoning"])
        panels.append(alt.layer(start, line, pts).properties(title=f"Stock {s}", width=260,
                                                             height=220))
    rows = [alt.hconcat(*panels[i:i + 3]) for i in range(0, len(panels), 3)]
    return alt.vconcat(*rows).resolve_scale(color="shared", shape="shared")


def estimates_chart(d: pd.DataFrame, group: str, order: list[str], xtitle: str,
                    groups: list[str]) -> alt.Chart:
    """Point estimates with 95% CIs, one row per `label`, one offset point per `group` (model
    or clustering). `d` has label, estimate, ci_lo, ci_hi, p and the group column."""
    palette = SERIES + [INK_2, "#9b59b6", "#c49a00"]
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12,
                                                               labelLimit=220))
    off = alt.YOffset(f"{group}:N", sort=groups)
    color = alt.Color(f"{group}:N", sort=groups, title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=groups, range=palette[:len(groups)]))
    tip = ["label", group, alt.Tooltip("estimate:Q", format="+.4f"),
           alt.Tooltip("ci_lo:Q", format="+.4f", title="CI low"),
           alt.Tooltip("ci_hi:Q", format="+.4f", title="CI high"),
           alt.Tooltip("p:Q", format=".4f")]
    base = alt.Chart(d).encode(y=y, yOffset=off, color=color, tooltip=tip)
    pts = base.mark_point(filled=True, size=60).encode(x=alt.X("estimate:Q", title=xtitle))
    err = base.mark_rule(strokeWidth=1.8).encode(x="ci_lo:Q", x2="ci_hi:Q")
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=INK_2).encode(x="v:Q")
    h = max(160, len(order) * (12 * len(groups) + 12))
    return alt.layer(zero, err, pts).properties(height=h)
