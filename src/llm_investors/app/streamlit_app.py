"""Viewer only: reads logs/ and results/; never runs simulations or calls the API.

    streamlit run src/llm_investors/app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, if not pip-installed
from llm_investors.agents.personas import PERSONAS  # noqa: E402
from llm_investors.paths import ROOT  # noqa: E402

st.set_page_config(page_title="LLM investor agents", layout="wide")


# --------------------------------------------------------------------------- data access
def read_log(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


@st.cache_data(show_spinner=False)
def cached_log(path: str, mtime: float) -> list[dict]:
    return read_log(Path(path))


def log_events(path: Path) -> list[dict]:
    return cached_log(str(path), path.stat().st_mtime)


def datasets(study: str) -> list[str]:
    names = [p.name for p in (ROOT / "logs").glob(f"{study}*") if p.is_dir()]
    names += [p.name for p in (ROOT / "results").glob(f"{study}*") if p.is_dir()]
    return sorted(set(names), key=lambda n: (n != study, n)) or [study]


def csv(name: str, file: str) -> pd.DataFrame | None:
    p = ROOT / "results" / name / file
    return pd.read_csv(p) if p.exists() else None


def cfg(study: str) -> dict:
    return yaml.safe_load((ROOT / "config" / f"{study}.yaml").read_text())


# --------------------------------------------------------------------------- sidebar
study = st.sidebar.radio("Study", ["study1", "study2"],
                         format_func=lambda s: {"study1": "Study 1: exogenous prices",
                                                "study2": "Study 2: sandbox market"}[s])
name = st.sidebar.selectbox("Data", datasets(study),
                            help="study*_dummy = rule-based agents (pipeline check), "
                                 "study*_dryrun = prompts only")
page = st.sidebar.radio("Page", ["Research setup", "Assumptions", "Observability", "Live view",
                                 "Findings"])
C = cfg(study)
spend = ROOT / "cache" / study / "spend.json"
if spend.exists():
    s = json.loads(spend.read_text())
    st.sidebar.metric("Spent (all runs of this study)", f"${s['spent_usd']:.4f}",
                      help=f"Hard cap ${s['cap_usd']:.2f}")
st.sidebar.caption(f"Model: {C['llm']['model']} · tier {C['llm']['service_tier']} · "
                   f"effort {C['llm']['reasoning_effort']}")

# --------------------------------------------------------------------------- pages
if page == "Research setup":
    st.title("LLM investor agents, memory, and market outcomes")
    st.markdown("""
**Study 1 (micro).** Agents trade five anonymized stocks on exogenous price paths.
*Does the memory architecture change the agents' disposition effect (DE = PGR − PLR)?*

**Study 2 (macro).** The same agents trade one stock with each other; prices come from a call
auction of their orders. *Does individual behavior aggregate into bubbles, underreaction or
volatility, and how do Taiwan-relevant market rules change them?*

Persona and memory prompts are identical in both studies.
""")
    st.subheader("Memory architectures")
    st.table(pd.DataFrame({
        "architecture": ["none", "log", "reflect"],
        "what the agent sees when deciding": [
            "current-period information only (prices, holdings, cash, average cost)",
            "+ factual history of all past periods (value, % change, trades, realized P&L vs cost)",
            "+ 3 lessons the agent wrote itself after periods 5 and 10 (latest replace earlier)"]}))
    st.subheader("Personas")
    for k, v in PERSONAS.items():
        st.markdown(f"**{k}** — {v}")
    st.subheader("Paired design (Study 1)")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];
  S [label="100 price scenarios\\n(5 stocks x 15 months,\\nseed = base + k)"];
  P [label="3 personas"];
  X [label="same persona x scenario\\n= matched set", shape=ellipse];
  N [label="none"]; L [label="log"]; R [label="reflect"];
  D [label="DE per run\\n(PGR - PLR)"];
  S -> X; P -> X; X -> N; X -> L; X -> R; N -> D; L -> D; R -> D;
}""")
    st.caption("Memory is assigned by design while persona and price path are held fixed, so "
               "differences in DE within a matched set are attributable to memory.")

elif page == "Assumptions":
    st.title("Assumptions and limitations")
    if study == "study1":
        m, pf = C["market"], C["portfolio"]
        st.markdown(f"""
- **Price model:** driftless log-price random walk, ln P(t+1) = ln P(t) + μ + σ·Z with
  μ = {m['mu']:g}, σ = {m['sigma']:g} per period; {m['n_stocks']} stocks × {m['periods']}
  periods, 1 period = 1 month, every path starts at {m['start_price']:g}. With μ = 0 the median
  price stays at the start price, so gains and losses are equally likely. No real data is used.
- **Prompt design {C.get('design', 'v1')}:** {"each stock's prices in the last "
  + str(m.get('price_history', 0)) + " months are shown to every arm (market information, not "
  "memory); the rules describe a neutral monthly review" if m.get('price_history') else
  "today's prices only; the rules say the agent has no other information"}.
- **Reference price:** average purchase price; endowment = {pf['initial_shares']} shares of
  each stock bought at {m['start_price']:g} plus {pf['initial_cash']:,.0f} cash.
- **Execution:** at today's price, sells before buys, no fees; infeasible orders are clipped
  and logged as `order_rejected`.
- **DE:** Odean's PGR/PLR on sale days only (all-periods variant also saved).
""")
    else:
        m, b = C["market"], C["baseline"]
        st.markdown(f"""
- **Market:** Smith–Suchanek–Williams (1988) design, one stock, {m['periods']} periods,
  dividends {m['dividends']} with equal probability; FV_t = 24 × remaining periods.
- **Agents:** {b['n_agents']} per market, equal wealth {m['wealth']:,} at FV₁ with
  {m['endowment_shares']} shares (cash = wealth − shares × FV₁); endowed shares are booked at
  FV₁ as their purchase price.
- **Call auction:** maximize volume; ties broken by
  {"smallest order imbalance (default; departs from the proposal, see README), then "
   if m.get('min_imbalance_tiebreak', True) else ""}closest to the previous price, then
  the lower price; pro-rata fills on the long side. If nothing crosses, the previous price is
  carried forward (shown as "no trade").
- **Treatments** vary across market runs, never within one; replicate *r* uses the same
  dividend draws and endowments in every cell (common random numbers).
- **DE inside the market** uses all periods with a position (one stock, so sale days have no
  paper counterpart).
""")
    st.markdown("""
**Limitations.** Pretrained knowledge (no tickers or dates are shown, but the model knows
finance); prompt sensitivity (results are conditional on these exact prompts); one base model for
all agents; temperature is not set (API default), so identical prompts give identical cached
replies unless `replicate_id` is changed; dummy datasets (`*_dummy`) come from rule-based agents
and are pipeline checks, not findings.
""")

elif page == "Observability":
    st.title("Observability")
    fatal_file = ROOT / "logs" / name / "FATAL_ERROR.json"
    if fatal_file.exists():
        f = json.loads(fatal_file.read_text())
        e = f["error"]
        st.error(f"**The last run was stopped by a {'quota / billing / spend-limit' if e['kind'] == 'quota' else 'non-retryable API'} error.** "
                 f"{e['error_type']} (HTTP {e['status']}, code {e['code']}): {e['message']}  \n"
                 f"{f['runs_complete']} runs complete, {len(f['runs_stopped_by_error'])} stopped "
                 f"by the error, {f['runs_aborted']} aborted. Request id: {e['request_id']}")
    logs = sorted((ROOT / "logs" / name).glob("run_*.jsonl"))
    if not logs:
        st.info(f"No logs in logs/{name}/ yet.")
    else:
        counts, rows = {}, []
        bar = st.progress(0.0, text="reading logs")
        for i, p in enumerate(logs):
            ev = log_events(p)
            start = ev[0] if ev else {}
            calls = [e for e in ev if e["type"] == "llm_call"]
            for e in ev:
                counts[e["type"]] = counts.get(e["type"], 0) + 1
            rows.append({
                "condition": (f"{start.get('architecture')} × {start.get('persona')}"
                              if study == "study1" else start.get("cell")),
                "runs": 1, "llm_calls": len(calls),
                "invalid": sum(not e["valid"] for e in calls),
                "fallbacks": sum(e["type"] == "fallback" for e in ev),
                "api_retries": sum(e.get("api_retries", 0) for e in calls),
                "incomplete": sum(e.get("incomplete_retries", 0) for e in calls),
                "incomplete_stops": sum(e["type"] == "incomplete_stop" for e in ev),
                "omitted_stock": sum(str(e.get("error") or "").startswith("missing an entry")
                                     for e in calls),
                "cache_hits": sum(e["cached"] for e in calls),
                "input_tokens": sum((e.get("usage") or {}).get("input_tokens", 0) for e in calls),
                "output_tokens": sum((e.get("usage") or {}).get("output_tokens", 0) for e in calls),
                "reasoning_tokens": sum((e.get("usage") or {}).get("reasoning_tokens", 0)
                                        for e in calls),
                "cost_usd": sum(e.get("cost_usd", 0) for e in calls)})
            bar.progress((i + 1) / len(logs), text=f"reading logs ({i + 1}/{len(logs)})")
        bar.empty()
        st.subheader("Event counts")
        st.dataframe(pd.DataFrame(sorted(counts.items()), columns=["event", "count"]),
                     hide_index=True)
        df = pd.DataFrame(rows).groupby("condition").sum(numeric_only=True)
        df["invalid_rate"] = df["invalid"] / df["llm_calls"]
        df["fallback_rate"] = df["fallbacks"] / df["llm_calls"]
        # share of API attempts that came back incomplete (retried, never held)
        df["incomplete_rate"] = df["incomplete"] / (df["llm_calls"] + df["incomplete"]).clip(lower=1)
        st.subheader("By condition")
        st.dataframe(df.style.format({"invalid_rate": "{:.2%}", "fallback_rate": "{:.2%}",
                                      "incomplete_rate": "{:.2%}", "cost_usd": "${:.4f}"}))
        t = df.sum()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("LLM calls", f"{int(t['llm_calls']):,}")
        c2.metric("Invalid replies", f"{t['invalid'] / max(1, t['llm_calls']):.2%}")
        c3.metric("Fallbacks", int(t["fallbacks"]))
        c4.metric("Incomplete (retried)", int(t["incomplete"]),
                  help=f"{int(t['incomplete_stops'])} run(s) stopped after repeated incompletes")
        c5.metric("Cost in these logs", f"${t['cost_usd']:.4f}")

elif page == "Live view":
    st.title("Live view")
    logs = sorted((ROOT / "logs" / name).glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        st.info(f"No logs in logs/{name}/ yet. Start a run: run-{study}")
        st.stop()
    follow = st.toggle("Follow the most recently updated run", value=True)
    choice = logs[-1] if follow else st.selectbox(
        "Run", logs[::-1], format_func=lambda p: p.stem.removeprefix("run_"))

    @st.fragment(run_every=2 if follow else None)
    def live(path: Path):
        ev = read_log(path)  # not cached: the file grows while the run is active
        start = ev[0]
        ends = [e for e in ev if e["type"] == "period_end"]
        done = any(e["type"] == "run_end" for e in ev)
        cost = sum(e.get("cost_usd", 0) for e in ev if e["type"] == "llm_call")
        T = start.get("periods") or len(start.get("prices", [])) - 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Run", start["run_id"])
        c2.metric("Period", f"{len(ends)}/{T}", help="finished" if done else "running")
        c3.metric("Run cost", f"${cost:.4f}")
        if study == "study2" and ends:
            c4.metric("Price / FV", f"{ends[-1]['price']:.1f} / {ends[-1]['fundamental_value']:.0f}")
        if not ends:
            st.info("Waiting for the first period…")
            return
        t = st.slider("Period", 1, len(ends), len(ends)) if len(ends) > 1 else 1
        pe = ends[t - 1]
        if study == "study1":
            prices = pd.DataFrame(start["prices"], columns=start["labels"])
            st.line_chart(prices.iloc[: t + 1], height=220, x_label="period", y_label="price")
        else:
            auc = [e for e in ev if e["type"] == "auction_cleared"][:t]
            st.line_chart(pd.DataFrame({"price": [a["price"] for a in auc],
                                        "fundamental value": [a["fundamental_value"] for a in auc]},
                                       index=range(1, len(auc) + 1)), height=220,
                          x_label="period")
        st.subheader(f"Decisions in period {t}")
        orders = pe["orders"]
        od = (pd.DataFrame(orders) if isinstance(orders, list)
              else pd.DataFrame([{"agent": a, **o} for a, o in orders.items()]))
        st.dataframe(od, hide_index=True, width="stretch")
        st.subheader("Prompt and raw-response inspector")
        calls = [e for e in ev if e["type"] == "llm_call" and e["period"] == t]
        if calls:
            labels = [f"{e['agent']} · {e['kind']} · attempt {e['attempt']}"
                      f"{' · INVALID' if not e['valid'] else ''}{' · cached' if e['cached'] else ''}"
                      for e in calls]
            i = st.selectbox("Call", range(len(calls)), format_func=lambda k: labels[k])
            e = calls[i]
            a, b = st.columns(2)
            with a:
                st.caption("System prompt")
                st.code(e["system"], language=None, wrap_lines=True)
                for m in e["messages"]:
                    st.caption(m["role"])
                    st.code(m["content"], language=None, wrap_lines=True)
            with b:
                st.caption(f"Raw response · status {e['status']} · "
                           f"{e['latency_s']:.1f}s · ${e['cost_usd']:.6f} · "
                           f"API retries {e['api_retries']}")
                st.code(e["raw"] or "(empty)", language="json", wrap_lines=True)
                if e.get("error"):
                    st.error(e["error"])
                st.caption("Usage")
                st.json(e["usage"] or {}, expanded=False)
    live(choice)

elif page == "Findings":
    st.title("Findings")
    if "dummy" in name or "dryrun" in name:
        st.warning("This dataset comes from rule-based dummy agents: it checks the pipeline "
                   "and is not a research finding.")
    an = ROOT / "results" / name / "analysis"
    if not an.exists():
        st.info(f"No analysis yet. Run: build-dataset {name} && "
                f"analyze-{study} {name}")
        st.stop()
    figs = an / "figures"
    if study == "study1":
        boot = csv(name, "analysis/de_bootstrap.csv")
        if (figs / "de_by_architecture.png").exists():
            st.image(str(figs / "de_by_architecture.png"))
        pooled = csv(name, "analysis/de_pooled.csv")
        if pooled is not None:
            st.subheader("Pooled PGR / PLR / DE with 95% CIs")
            st.caption("Cluster bootstrap over matched sets (persona × scenario). The second "
                       "sample drops the final period, when agents know the game ends.")
            st.dataframe(pooled.round(3), hide_index=True)
        st.subheader("Run-level DE by architecture × persona")
        st.caption("mean_run_DE averages runs over matched sets where DE is defined in every "
                   "arm; n_sets_dropped counts sets excluded for that reason.")
        st.dataframe(boot.round(3), hide_index=True)
        st.subheader("Memory contrasts (matched sets, 95% cluster-bootstrap CI)")
        st.dataframe(csv(name, "analysis/de_differences.csv").round(3), hide_index=True)
        fr = json.loads((an / "friedman.json").read_text())
        if fr:
            st.caption(f"Friedman test across architectures: χ² = {fr['statistic']:.2f}, "
                       f"p = {fr['p_value']:.3f}, {fr['n_sets']} matched sets")
        st.subheader("Regressions (SEs clustered by matched set: persona × scenario)")
        st.code((an / "regressions.txt").read_text(), language=None)
        st.subheader("Sample paths with sales marked")
        for p in sorted(figs.glob("paths_*.png")):
            st.image(str(p))
    else:
        for f in ("price_vs_fv.png", "outcomes_by_cell.png"):
            if (figs / f).exists():
                st.image(str(figs / f))
        st.subheader("Outcomes by treatment (descriptive)")
        st.caption("3–5 market runs per cell: mean, sd, min and max, no confidence intervals.")
        st.dataframe(csv(name, "analysis/outcomes_by_cell.csv").round(3), hide_index=True)
        paired = csv(name, "analysis/paired_vs_baseline.csv")
        if paired is not None and not paired.empty:
            st.subheader("Paired differences vs baseline (same replicate)")
            st.caption("Replicate r uses the same dividend draws and endowments in every cell, "
                       "so cell − baseline removes that shared noise.")
            st.dataframe(paired.round(3), hide_index=True)
        heat = sorted(figs.glob("actions_heatmap_*.png"))
        if heat:
            st.subheader("Orders by agent and period (first replicate of each cell)")
            pick = st.selectbox("Cell", heat,
                                format_func=lambda p: p.stem.removeprefix("actions_heatmap_"))
            st.image(str(pick))
        st.subheader("Disposition effect inside the market")
        summ = json.loads((an / "summary.json").read_text())
        st.warning(summ.get("agent_de_note", "Not comparable like for like with Study 1."))
        st.dataframe(csv(name, "analysis/agent_de.csv").round(3), hide_index=True)
        st.subheader("Stylized facts")
        st.json(json.loads((an / "stylized_facts.json").read_text()))
