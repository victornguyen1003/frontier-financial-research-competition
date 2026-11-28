"""Viewer for Study 1 design v3 (plan §6). Reads files only; never runs a paid simulation or
calls the API. The prompt builder replays a rule-based run in memory to show exact prompts.

logs/ is git-ignored, so a deployed copy (a fresh git checkout) reads the gzipped copies in
published/logs/ instead (`publish-logs`).

    investor-viewer          # or: streamlit run src/llm_investors/app/streamlit_app.py
"""
from __future__ import annotations

import difflib
import gzip
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, if not pip-installed
from llm_investors.agents import personas as PZ  # noqa: E402
from llm_investors.agents import prompts as PR  # noqa: E402
from llm_investors.app import charts as CH  # noqa: E402
from llm_investors.paths import ROOT  # noqa: E402

st.set_page_config(page_title="LLM investor personas", layout="wide")
PUBLISHED = ROOT / "published" / "logs"
ARCHS = ["none", "log", "reflect"]
TAGS = ["persona", "reference_point", "past_trades", "forecast", "randomness"]


# --------------------------------------------------------------------------- helpers
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "study1.yaml").read_text())


def read_log(path: Path) -> list[dict]:
    text = (gzip.decompress(path.read_bytes()).decode("utf-8") if path.suffix == ".gz"
            else path.read_text(encoding="utf-8"))
    out = []
    for line in text.splitlines():
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


def log_dir(name: str) -> tuple[Path, bool]:
    local = ROOT / "logs" / name
    if any(local.glob("run_*.jsonl")) or not (PUBLISHED / name).is_dir():
        return local, False
    return PUBLISHED / name, True


def log_files(name: str) -> list[Path]:
    d, pub = log_dir(name)
    return sorted(d.glob("run_*.jsonl.gz" if pub else "run_*.jsonl"))


def run_label(p: Path) -> str:
    return p.name.split(".")[0].removeprefix("run_")


def datasets() -> list[str]:
    # dry runs only build prompts for a cost estimate: never shown as datasets
    names = {p.name for base in (ROOT / "logs", PUBLISHED, ROOT / "results")
             for p in base.glob("study1_v3*") if p.is_dir()
             and not p.name.endswith(("__replay", "_dryrun"))}

    def rank(n):
        return (n.endswith("_dummy"), "pilot" in n, n)
    return sorted(names, key=rank)


def table(name: str, file: str) -> pd.DataFrame | None:
    d = ROOT / "results" / name
    for p in (d / f"{file}.parquet", d / f"{file}.csv"):
        if p.exists():
            try:
                return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            except Exception:  # noqa: BLE001
                continue
    return None


def jload(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def show(df: pd.DataFrame | None, note: str = "Not available yet."):
    if df is None or df.empty:
        st.caption(note)
    else:
        st.dataframe(df.round(4), hide_index=True)


def image(p: Path, caption: str | None = None):
    if p.exists():
        st.image(str(p))
        if caption:
            st.caption(caption)
    else:
        st.caption(f"({p.relative_to(ROOT)} not produced yet)")


def md_section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else f"(section '{heading}' not found in {path.name})"


def tokens(s: str) -> int:
    return max(1, len(s) // 4)


def spend() -> dict:
    p = ROOT / "cache" / "study1" / "spend.json"
    s = jload(p) or {}
    paid = 0.0
    for f in (ROOT / "cache" / "study1").glob("??/*.json"):
        try:
            paid += float(json.loads(f.read_text()).get("paid_usd", 0))
        except (json.JSONDecodeError, ValueError):
            pass
    return {"ledger": s.get("spent_usd", 0.0), "cache": paid}


# --------------------------------------------------------------------------- sidebar
C = cfg()
all_ds = datasets()
name = st.sidebar.selectbox("Dataset", all_ds or ["(none yet)"],
                            help="study1_v3_<model>_pilot = pilot; *_dummy = rule-based agents "
                                 "(pipeline check, not findings)")
page = st.sidebar.radio("Page", ["Research goal and setup", "Agent prompts", "Live view",
                                 "Final dataset", "Data analysis", "Findings and conclusions"])
sp = spend()
st.sidebar.metric("Spent (prototype 2)", f"${max(sp['ledger'], sp['cache']):.4f}",
                  help=f"Hard cap ${C['budget']['cap_usd']:.2f}")
st.sidebar.caption(f"Main model {C['llm']['model']} · tier {C['llm']['service_tier']} · "
                   f"effort {C['llm']['reasoning_effort']} · design {C['design']}")
if "dummy" in name:
    st.sidebar.warning("Dummy dataset: rule-based agents, not findings.")


# =========================================================================== 1. setup
if page == "Research goal and setup":
    st.title("Do customer profiles change an LLM investor's disposition effect?")
    st.markdown("""
- **RQ1:** do persona attributes change PGR, PLR and PGR − PLR compared with the control persona C0?
- **RQ2:** which attributes move the disposition effect (DE) and by how much, and which are blind spots?
- **RQ3:** does memory (`log`, `reflect`) change the DE, and does the reasoning refer to past trades?
""")
    st.subheader("Benchmark: Barber et al. (2007), Table 3 Panel B")
    bm = pd.read_csv(ROOT / "data" / "benchmarks" / "barber_2007_table3_panelB.csv")
    st.dataframe(bm.assign(PGR=bm["PGR"].map("{:.2%}".format), PLR=bm["PLR"].map("{:.2%}".format)),
                 hide_index=True)
    st.subheader("Run grid")
    a = C["agents"]["architectures"]
    rows = []
    for stg, s in C["stages"].items():
        n = (s["profiles"] + 2) * s["scenarios"] * len(a)
        rows.append({"stage": stg, "profiles P": s["profiles"], "fixed": "C0, C1",
                     "paths S": s["scenarios"], "arms": ", ".join(a), "runs": n})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Every persona runs on the same S paths in all 3 arms; scenario-major order, so "
               "a budget stop removes whole matched sets. Pilots run once per model.")

    st.subheader("Pilot comparison")
    pil = [n for n in all_ds if n.endswith("_pilot")]
    prow = []
    for n in pil:
        proj = jload(ROOT / "results" / n / "pilot_projection.json") or {}
        t = jload(ROOT / "results" / n / "analysis" / "tests.json") or {}
        runs = table(n, "runs")
        mc = t.get("manipulation_check") or {}
        rq1 = t.get("rq1_paired") or {}
        inc = None
        if runs is not None and not runs.empty:
            inc = runs["incomplete"].sum() / max(1, runs["llm_calls"].sum() + runs["incomplete"].sum())
        prow.append({"pilot": n, "cost per run $": proj.get("cost_per_run_usd"),
                     "incomplete rate": inc,
                     "manipulation DE(C1)−DE(C0)": mc.get("mean_diff"),
                     "manipulation p (one-sided)": mc.get("p_wilcoxon"),
                     "sd of profile DE − C0": rq1.get("spread_sd_of_profile_minus_C0"),
                     "projected main run $": proj.get("projected_full_cost_usd")})
    show(pd.DataFrame(prow), "No pilot yet: run-study1 --stage pilot --model gpt-6-luna "
                             "(and --model gpt-6.2-sol).")

    st.subheader("Power")
    pw = [n for n in all_ds if (ROOT / "results" / n / "power" / "recommendation.json").exists()]
    if pw:
        pick = st.selectbox("Pilot", pw, key="power_pilot")
        rec = jload(ROOT / "results" / pick / "power" / "recommendation.json")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended P × S", f"{rec['P']} × {rec['S']}", help=rec["reason"])
        c2.metric("Runs", f"{rec['runs']:,}",
                  help=None if rec["projected_cost_usd"] is None
                  else f"≈ ${rec['projected_cost_usd']:.2f}")
        c3.metric("MDE of d_k (Holm, 80%)", f"{rec['mde_dk_holm']:.3f}")
        image(ROOT / "results" / pick / "power" / "power_curves.png")
        show(pd.read_csv(ROOT / "results" / pick / "power" / "mde.csv"))
    else:
        st.caption("Not run yet: power-study1 <pilot dataset>.")

    st.subheader("Market parameters")
    m, pf = C["market"], C["portfolio"]
    st.markdown(f"""
- {m['n_stocks']} anonymous stocks (A–E), {m['periods']} monthly periods, all start at {m['start_price']:.0f}
- Prices: zero-drift log random walk; each stock draws its monthly σ from the TEJ distribution, Student-t shocks (`{m['generator']}`); path k uses seed {C['seed']} + k
- Endowment: {pf['initial_cash']:,.0f} cash + {pf['initial_shares']} shares of each stock bought at {m['start_price']:.0f}
- Market orders at today's price; no fees, taxes, borrowing or short sales
- Every arm sees the account, average cost and the full price history since month 0
""")
    st.subheader("Data and calibration")
    pj = jload(ROOT / "results" / "calibration" / "prices.json")
    aj = jload(ROOT / "results" / "calibration" / "attributes.json")
    if pj:
        s, t, cov = pj["sigma_summary"], pj["student_t"], pj["coverage"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks used", cov["stocks_used"])
        c2.metric("Monthly σ (median)", f"{s['median']:.3f}", help=f"IQR {s['p25']:.3f}–{s['p75']:.3f}")
        c3.metric("Student-t df", f"{t['df']:.1f}", help=t["method"])
        c4.metric("Period", f"{cov['first_date']} – {cov['last_date']}")
        if cov.get("months_excluded_incomplete"):
            st.warning(f"Months only partly covered by the TEJ file and excluded: "
                       f"{', '.join(cov['months_excluded_incomplete'])}.")
        g = ROOT / "results" / "calibration" / "figures"
        c1, c2 = st.columns(2)
        with c1:
            image(g / "sigma_hist.png")
        with c2:
            image(g / "paths_real_vs_sim.png")
        image(g / "returns_qq.png")
        st.markdown("**Random-walk tests (daily log returns)**")
        rw = pj["random_walk_tests"]
        st.dataframe(pd.DataFrame(
            [{"test": f"autocorrelation lag {k[-1]}", "per-stock median": v["median"],
              "index": v["index"]} for k, v in rw["autocorrelation"].items()]
            + [{"test": f"Ljung-Box {k}: share p<0.05", "per-stock median":
                v["share_p_below_0.05"], "index": v["index_p"]} for k, v in rw["ljung_box"].items()]
            + [{"test": f"variance ratio {k}", "per-stock median": v["median_vr"],
                "index": v["index_vr"]} for k, v in rw["variance_ratio"].items()]).round(3),
            hide_index=True)
        st.caption(rw["note"])
    else:
        st.caption("Not calibrated yet: run `calibrate`.")
    if aj:
        st.markdown(f"**Age levels:** younger {aj['younger_age']}, older {aj['older_age']}. "
                    f"{aj['method']}")
        br = pd.DataFrame(aj["tdcc_age_distribution"]["brackets"])
        st.dataframe(br, hide_index=True)
        url = aj["tdcc_age_distribution"]["source_url"]
        st.caption(f"Source: {aj['tdcc_age_distribution']['title']}"
                   + (f" ({url})" if url else " (URL to be added in config)"))

    st.subheader("Attribute catalog")
    st.dataframe(pd.DataFrame([{"#": i + 1, "key": k, "level 1": a1, "level 0": a0 or "—"}
                               for i, (k, a1, a0) in enumerate(PZ.ATTRIBUTES)]), hide_index=True)
    try:
        ages = PZ.load_ages()
        st.markdown(f"**C0 (control):** {PZ.compose_persona(PZ.control_profile(), ages)}")
        st.markdown(f"**C1 (bias cue):** {PZ.compose_persona(PZ.control_profile(), ages, cue=True)}")
    except FileNotFoundError:
        st.caption("C0/C1 texts need results/calibration/attributes.json.")
    st.markdown("""**Memory arms**
- `none`: nothing beyond the public state
- `log`: the agent's own record, one line per past month (account value, % change, trades, realized gain/loss vs average cost)
- `reflect`: the `log` record plus three lessons written by a reflection call after months 5 and 10
""")
    notes = ROOT / "docs" / "prototype-2-research-notes.md"
    with st.expander("Assumptions"):
        st.markdown(md_section(notes, "Assumptions"))
    with st.expander("Limitations"):
        st.markdown(md_section(notes, "Limitations"))
    with st.expander("Gaps found during implementation"):
        st.markdown(md_section(notes, "Gaps found during implementation"))
    st.subheader("Budget")
    lim = C["budget"].get("stage_limits") or {}
    st.dataframe(pd.DataFrame([{"stage": k, "limit $": v} for k, v in lim.items()]
                              + [{"stage": "TOTAL cap", "limit $": C["budget"]["cap_usd"]}]),
                 hide_index=True)
    st.metric("Spent so far", f"${max(sp['ledger'], sp['cache']):.4f}",
              help="Larger of this machine's ledger and the total recorded in cache/study1/")

# =========================================================================== 2. prompts
elif page == "Agent prompts":
    st.title("Agent prompts")
    from llm_investors.envs import exogenous as X
    from llm_investors.sim import run_study1 as R
    try:
        ages = PZ.load_ages()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        kind = st.radio("Persona", ["C0", "C1", "Custom attributes", "Pilot profile"],
                        horizontal=True)
    with c2:
        arch = st.selectbox("Memory arm", ARCHS, index=2)
    with c3:
        month = st.slider("Month", 1, C["market"]["periods"], 7)
    prof = PZ.control_profile()
    pid = kind
    if kind == "Custom attributes":
        cols = st.columns(4)
        for i, (k, _, _) in enumerate(PZ.ATTRIBUTES):
            prof[k] = int(cols[i % 4].checkbox(PZ.LABELS[k], key=f"a_{k}"))
        pid = "custom"
    elif kind == "Pilot profile":
        roster = PZ.roster(C["stages"]["pilot"]["profiles"], C["profiles"]["seed"], ages)
        pick = st.selectbox("Profile", [r["id"] for r in roster if r["kind"] == "profile"])
        prof = next(r for r in roster if r["id"] == pick)["profile"]
        pid = pick
    text = PZ.compose_persona(prof, ages, cue=kind == "C1")
    persona = {"id": pid, "kind": "C1" if kind == "C1" else ("C0" if kind == "C0" else "profile"),
               "profile": prof, "text": text, "n_active": sum(prof.values())}

    st.subheader("Persona against C0")
    c0 = PZ.compose_persona(PZ.control_profile(), ages)
    split = lambda s: re.split(r"(?<=\.) ", s)
    a_, b_ = split(c0), split(text)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_, b_).get_opcodes():
        if op == "equal":
            out += a_[i1:i2]
        else:
            out += [f"~~{x}~~" for x in a_[i1:i2]] + [f"**:blue[{x}]**" for x in b_[j1:j2]]
    st.markdown(" ".join(out))
    st.caption("Blue: sentences that differ from C0; struck through: C0 sentences replaced.")

    class _Echo:
        dry_run = False

        def complete(self, system, messages, dummy=None):
            return {"text": dummy(), "status": "dummy", "cached": False, "key": "x", "usage": {},
                    "latency_s": 0.0, "cost_usd": 0.0, "api_retries": 0}

    class _Board:
        def update(self, *a, **k): pass
        def finish(self, *a, **k): pass
        def echo(self, *a, **k): pass

    cfg_dummy = {**C, "llm": {**C["llm"], "backend": "dummy"}}
    tmp = Path(tempfile.mkdtemp(prefix="viewer_"))
    R.run_one(cfg_dummy, _Echo(), "prompt", persona, arch, 0, threading.Event(), _Board(),
              log_root=tmp)
    ev = read_log(next((tmp / "prompt").glob("run_*.jsonl")))
    calls = [e for e in ev if e["type"] == "llm_call"]
    dec = next(e for e in calls if e["kind"] == "decision" and e["period"] == month
               and e["attempt"] == 0)
    st.caption("Path 0. Earlier months use rule-based replies, so the record and lessons are "
               "examples; the layout and every fixed block are exact.")
    a, b = st.columns(2)
    with a:
        st.markdown("**System prompt**")
        st.code(dec["system"], language=None, wrap_lines=True)
    with b:
        st.markdown("**User prompt**")
        st.code(dec["messages"][0]["content"], language=None, wrap_lines=True)
    user = dec["messages"][0]["content"]
    blocks = {"rules": X.rules(C["design"]), "persona": "Who you are: " + text,
              "reply format": X.env_spec(5, C["design"]).order_format}
    mem, _, rest = user.partition("Period ")
    blocks["memory"] = mem
    state, _, hist = ("Period " + rest).partition("Price history:")
    blocks["state"] = state
    blocks["price history"] = "Price history:" + hist
    st.markdown("**Approximate tokens per block** (characters / 4)")
    st.dataframe(pd.DataFrame([{"block": k, "tokens": tokens(v) if v.strip() else 0}
                               for k, v in blocks.items()]), hide_index=True)
    st.subheader("Reflection prompt (reflect arm, after months 5 and 10)")
    refl = [e for e in calls if e["kind"] == "reflection"]
    if refl:
        a, b = st.columns(2)
        with a:
            st.code(refl[0]["system"], language=None, wrap_lines=True)
        with b:
            st.code(refl[0]["messages"][0]["content"], language=None, wrap_lines=True)
    else:
        st.caption("Choose the reflect arm.")
    st.subheader("Reply formats")
    st.code(X.env_spec(5, C["design"]).order_format, language=None)
    st.code(PR.REFLECT_FORMAT, language=None)
    st.caption("A reply needs a non-empty `reasoning` of at most 1,000 characters and one "
               "entry per stock; otherwise it gets one correction retry, then a logged hold.")

# =========================================================================== 3. live
elif page == "Live view":
    st.title("Live view")
    logs = log_files(name)
    published = log_dir(name)[1]
    if not logs:
        st.info(f"No logs for {name}. Start a run: run-study1 --stage pilot --model <id>")
        st.stop()

    @st.fragment(run_every=2)
    def monitor():
        files = log_files(name)
        now = time.time()
        stat = {"complete": 0, "running": 0, "stopped": 0}
        inc = fb = om = err = 0
        cost = 0.0
        for p in files:
            ev = read_log(p) if not published else log_events(p)
            end = next((e for e in ev if e["type"] == "run_end"), None)
            if end is None:
                stat["running" if now - p.stat().st_mtime < 120 else "stopped"] += 1
            elif end["status"] == "complete":
                stat["complete"] += 1
            else:
                stat["stopped"] += 1
            calls = [e for e in ev if e["type"] == "llm_call"]
            inc += sum(e.get("incomplete_retries", 0) for e in calls)
            fb += sum(e["type"] == "fallback" for e in ev)
            om += sum(str(e.get("error") or "").startswith("missing an entry") for e in calls)
            err += sum(e["type"] == "fatal_api_error" for e in ev)
            cost += sum(e.get("cost_usd", 0) for e in calls)
        c = st.columns(7)
        c[0].metric("Complete", stat["complete"])
        c[1].metric("Running", stat["running"])
        c[2].metric("Stopped", stat["stopped"])
        s = spend()
        c[3].metric("Spend vs cap", f"${max(s['ledger'], s['cache']):.3f}",
                    help=f"cap ${C['budget']['cap_usd']:.2f}; these logs ${cost:.4f}")
        c[4].metric("Incomplete (retried)", inc)
        c[5].metric("Fallbacks / omitted", f"{fb} / {om}")
        c[6].metric("API errors", err)
        fatal = log_dir(name)[0] / "FATAL_ERROR.json"
        if fatal.exists():
            f = json.loads(fatal.read_text())["error"]
            st.error(f"Stopped by {f['kind']} error: {f['error_type']} (HTTP {f['status']}): "
                     f"{f['message']}")
    monitor()

    st.subheader("Single run")
    files = sorted(logs, key=lambda p: p.stat().st_mtime) if not published else logs
    follow = (not published) and st.toggle("Follow the most recently updated run", value=False)
    choice = files[-1] if follow else st.selectbox("Run", files[::-1], format_func=run_label)

    @st.fragment(run_every=2 if follow else None)
    def single(path: Path):
        ev = read_log(path)
        start = ev[0]
        ends = [e for e in ev if e["type"] == "period_end"]
        T = len(start.get("prices", [])) - 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Run", start["run_id"])
        c2.metric("Month", f"{len(ends)}/{T}")
        c3.metric("Persona", f"{start.get('persona')} ({start.get('persona_kind')})")
        st.caption(start.get("persona_text", ""))
        if not ends:
            st.info("Waiting for the first month…")
            return
        t = st.slider("Month", 1, len(ends), len(ends)) if len(ends) > 1 else 1
        pe = ends[t - 1]
        a, b = st.columns([3, 2])
        with a:
            prices = pd.DataFrame(start["prices"], columns=start["labels"])
            st.line_chart(prices.iloc[: t + 1], height=240, x_label="month", y_label="price")
        with b:
            h = pd.DataFrame([{"stock": s, "shares": v["shares"],
                               "average cost": round(v["avg_cost"], 2),
                               "price": start["prices"][t][i]}
                              for i, (s, v) in enumerate(pe["holdings"].items())])
            st.dataframe(h, hide_index=True)
            st.caption(f"Cash {pe['cash']:,.2f} · account value {pe['value']:,.2f}")
        st.markdown("**Reasoning**")
        st.info(pe.get("reasoning") or "(none: fallback or design without reasoning)")
        st.markdown("**Orders**")
        st.dataframe(pd.DataFrame(pe["orders"]), hide_index=True)
        les = [e for e in ev if e["type"] == "reflection" and e["period"] < t]
        if les:
            st.markdown(f"**Lessons in force (written after month {les[-1]['period']})**")
            for i, x in enumerate(les[-1]["lessons"], 1):
                st.markdown(f"{i}. {x}")
        with st.expander("Prompt and raw response"):
            calls = [e for e in ev if e["type"] == "llm_call" and e["period"] == t]
            for e in calls:
                st.caption(f"{e['kind']} · attempt {e['attempt']} · status {e['status']}"
                           f"{' · INVALID: ' + e['error'] if not e['valid'] else ''}"
                           f"{' · cached' if e['cached'] else ''} · ${e['cost_usd']:.6f}")
                st.code(e["messages"][-1]["content"], language=None, wrap_lines=True)
                st.code(e["raw"] or "(empty)", language="json", wrap_lines=True)
    single(choice)

# =========================================================================== 4. dataset
elif page == "Final dataset":
    st.title("Final dataset")
    if "dummy" in name:
        st.warning("Rule-based dummy agents: a pipeline check, not findings.")
    tabs = {"runs": "runs", "decisions": "decisions", "positions": "positions",
            "lessons": "lessons", "reasoning_tags": "reasoning_tags", "reasoning": "reasoning"}
    which = st.radio("Table", list(tabs), horizontal=True)
    df = table(name, tabs[which])
    if df is None or df.empty:
        st.info(f"No {which} table for {name}. Run build-dataset {name}"
                + (f" and tag-reasoning {name}" if which == "reasoning_tags" else "") + ".")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    if "persona" in df:
        sel = c1.multiselect("Persona / profile", sorted(df["persona"].dropna().unique()))
        if sel:
            df = df[df["persona"].isin(sel)]
    if "architecture" in df:
        sel = c2.multiselect("Arm", ARCHS)
        if sel:
            df = df[df["architecture"].isin(sel)]
    if "scenario" in df:
        sel = c3.multiselect("Path", sorted(df["scenario"].dropna().unique()))
        if sel:
            df = df[df["scenario"].isin(sel)]
    attr = c4.selectbox("Attribute at level 1", ["(any)"] + PZ.ATTRIBUTE_KEYS)
    if attr != "(any)" and f"attr_{attr}" in df:
        df = df[df[f"attr_{attr}"] == 1]
    st.caption(f"{len(df):,} rows")
    st.dataframe(df, hide_index=True, height=420)
    st.download_button(f"Download {which}.csv (filtered)", df.to_csv(index=False).encode(),
                       f"{name}_{which}.csv", "text/csv")
    st.subheader("Column dictionary")
    st.dataframe(pd.DataFrame([
        ("run_id", "arm_persona_sNNN"), ("architecture", "memory arm: none, log, reflect"),
        ("persona", "P00.. (sampled profile), C0 (control), C1 (bias cue)"),
        ("persona_kind", "profile / C0 / C1"), ("n_active", "attributes at level 1"),
        ("attr_<key>", "attribute level (0/1), plan §2.2"), ("scenario", "price path k (seed + k)"),
        ("period", "month 1..15"), ("gain", "1 if price > average cost, 0 if below (equal: missing)"),
        ("sold / sell", "position sold this month"), ("ret", "price / average cost − 1"),
        ("holding_period", "months since the position was opened"),
        ("drawdown", "account value / running peak − 1"),
        ("sale_day", "1 if the run sold anything this month (Odean sample)"),
        ("RG, RL, PG, PL", "realized / paper gains and losses on sale days"),
        ("PGR, PLR, DE, Ratio", "RG/(RG+PG), RL/(RL+PL), PGR−PLR, PGR/PLR"),
        ("reasoning", "the reply's reasoning text"),
        ("actions", "action letter per stock A–E (B/S/H)"),
        ("tag_<tag>", "classifier label (1/0), plan §4.2"),
        ("invalid_replies, fallbacks, incomplete, omitted_stock_replies, reasoning_errors",
         "observability counts per run"),
    ], columns=["column", "meaning"]), hide_index=True)
    st.subheader("Data quality")
    runs = table(name, "runs")
    if runs is not None and not runs.empty:
        q = {"runs": len(runs), "complete": int(runs["complete"].sum()),
             "LLM calls": int(runs["llm_calls"].sum()),
             "invalid replies": int(runs["invalid_replies"].sum()),
             "fallbacks (holds)": int(runs["fallbacks"].sum()),
             "omitted-stock replies": int(runs["omitted_stock_replies"].sum()),
             "reasoning errors": int(runs.get("reasoning_errors", pd.Series([0])).sum()),
             "incomplete (retried)": int(runs["incomplete"].sum()),
             "incomplete stops": int(runs["incomplete_stops"].sum()),
             "rejected orders (clipped)": int(runs["orders_rejected"].sum()),
             "runs with DE undefined": int(runs["DE"].isna().sum()),
             "cost in logs $": round(float(runs["cost_usd"].sum()), 4)}
        st.dataframe(pd.DataFrame(q.items(), columns=["check", "value"]), hide_index=True)

# =========================================================================== 5. analysis
elif page == "Data analysis":
    st.title("Data analysis")
    if "dummy" in name:
        st.warning("Rule-based dummy agents: a pipeline check, not findings.")
    an = ROOT / "results" / name / "analysis"
    t = jload(an / "tests.json")
    if t is None:
        st.info(f"No analysis yet. Run: build-dataset {name} && analyze-study1 {name}")
        st.stop()
    figs = an / "figures"
    st.caption(f"{t['n_runs']} complete runs, {t['n_decision_rows']:,} sale-day position rows. "
               f"Regression: {t.get('regression_model')}")
    cells = table(name, "analysis/cells")
    runs = table(name, "runs")
    shown: list[str] = []
    if cells is not None and not cells.empty:
        present = set(cells["persona"])
        refs = [p for p in CH.CONTROLS if p in present]
        profs = sorted(p for p in present if p not in CH.CONTROLS)
        ranked = [p for p in CH.rank_by(cells) if p in profs]
        with st.sidebar:
            st.divider()
            st.markdown("**Personas in the charts**")
            st.caption("Used by the persona, ratio and memory charts below.")
            mode = st.radio("Profiles", ["None (reference rows only)", "Highest and lowest DE",
                                         "By attribute", "Choose"], index=1, key="pp_mode")
            chosen: list[str] = []
            if profs and mode == "Highest and lowest DE":
                half = max(1, len(ranked) // 2)
                n = st.slider("How many at each end", 1, half, min(5, half), key="pp_n")
                chosen = list(dict.fromkeys(ranked[:n] + ranked[-n:]))
            elif profs and mode == "By attribute":
                keys = [k for k in PZ.ATTRIBUTE_KEYS
                        if runs is not None and f"attr_{k}" in runs]
                if keys:
                    k = st.selectbox("Attribute", keys, format_func=lambda x: PZ.LABELS[x],
                                     key="pp_attr")
                    lvl = st.radio("Level", [1, 0], horizontal=True, key="pp_lvl",
                                   format_func=lambda v: f"level {v}")
                    att = runs.drop_duplicates("persona").set_index("persona")[f"attr_{k}"]
                    chosen = [p for p in profs if att.get(p) == lvl]
                    st.caption(f"{len(chosen)} profiles at level {lvl}.")
                else:
                    st.caption("No attribute columns in this dataset.")
            elif profs and mode == "Choose":
                chosen = st.multiselect("Profiles", profs, default=ranked[:3], key="pp_pick")
            ref_pick = st.multiselect("Reference rows", refs, default=refs, key="pp_refs",
                                      help="C0: no persona. C1: persona that states the "
                                           "mechanism. Pooled: all sampled profiles together.")
            sort = st.radio("Order profiles by", ["DE, all arms (high to low)", "Name"],
                            key="pp_sort")
            if sort == "Name":
                chosen = sorted(chosen)
            else:
                chosen = [p for p in ranked if p in set(chosen)]
            shown = [p for p in refs if p in ref_pick] + chosen

    st.header("Tests")
    rows = []
    mc = t.get("manipulation_check")
    if mc:
        rows.append({"test": "Manipulation check: DE(C1) > DE(C0)", "estimate": mc["mean_diff"],
                     "95% CI": f"[{mc.get('ci_lo', float('nan')):+.3f}, {mc.get('ci_hi', float('nan')):+.3f}]",
                     "p": mc.get("p_wilcoxon"), "MDE": mc.get("mde")})
    j = t.get("rq1_joint")
    if j:
        rows.append({"test": "RQ1 joint: all d_k = 0 (Wald F)", "estimate": j["F"], "95% CI": "",
                     "p": j["p"], "MDE": (t.get("rq2") or {}).get("mean_mde_holm")})
    rp = t.get("rq1_paired")
    if rp:
        rows.append({"test": "RQ1 paired: profiles differing from C0 (Holm)",
                     "estimate": rp["n_significant_holm"], "95% CI": "",
                     "p": None, "MDE": None})
    mw = t.get("rq3_memory_wald")
    if mw:
        rows.append({"test": "RQ3 memory: f_log = f_reflect = 0 (Wald F)", "estimate": mw["F"],
                     "95% CI": "", "p": mw["p"],
                     "MDE": max((v["mde"] for v in (t.get("memory_coefs") or {}).values()),
                                default=None)})
    for k, v in (t.get("memory_coefs") or {}).items():
        rows.append({"test": f"  {k}", "estimate": v["coef"], "95% CI": "", "p": v["p"],
                     "MDE": v["mde"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("MDE: smallest true effect detected with 80% power at this sample size (Holm "
               "level for d_k). A non-significant test with an MDE above the effect of interest "
               "is inconclusive, not evidence of no effect.")

    st.header("Cells (§5.1)")
    if cells is None or cells.empty:
        st.caption("Cells table not available yet.")
    elif not shown:
        st.caption("Choose personas in the sidebar.")
    else:
        c1, c2 = st.columns([1, 2])
        metric = c1.selectbox("Measure", list(CH.METRICS), key="cell_metric",
                              format_func=lambda m: CH.METRICS[m])
        arm_opts = ARCHS + ["all arms"]
        arms = c2.multiselect("Memory arms", arm_opts, default=ARCHS, key="cell_arms")
        bm = None
        if metric == "Ratio":
            bmall = pd.read_csv(ROOT / "data" / "benchmarks" / "barber_2007_table3_panelB.csv")
            which = st.multiselect("Benchmarks: Barber et al. (2007), Table 3 Panel B",
                                   list(bmall["investor_type"]),
                                   default=["Individuals", "Mutual funds"], key="cell_bm")
            bm = bmall[bmall["investor_type"].isin(which)]
        if not arms:
            st.caption("Choose at least one memory arm.")
        else:
            ch, dropped = CH.cells_chart(cells, shown, arms, metric, bm)
            if ch is not None:
                st.altair_chart(ch, width="stretch")
            if dropped:
                st.caption("Not drawn (ratio undefined because PLR = 0, i.e. no loser was "
                           "ever sold): " + ", ".join(dropped))
            cap = {"DE": "Bars right of zero: the persona sells winners more readily than "
                         "losers in that arm.",
                   "PGR": "Share of paper gains realized on sale days.",
                   "PLR": "Share of paper losses realized on sale days.",
                   "Ratio": "Points right of the 'Individuals 2.63' line show a stronger "
                            "winner/loser asymmetry than Taiwanese individuals in Barber et al.; "
                            "compare ratios and signs, not levels. Lower CI ends at or below 0 "
                            f"are drawn at {CH.RATIO_FLOOR}."}[metric]
            st.caption(cap + " Lines: 95% path-bootstrap CI. Hover for PGR, PLR and run counts.")
        with st.expander("Table for the personas and arms shown"):
            show(cells[cells["persona"].isin(shown) & cells["architecture"].isin(arms or ARCHS)]
                 .drop(columns=[c for c in ("kind",) if c in cells]))
        with st.expander("Full cells table"):
            show(cells)

    st.header("Attribute effects (§5.2, RQ2)")
    bs = table(name, "analysis/blind_spots")
    if bs is not None and not bs.empty and bs["coef"].notna().any():
        bsv = bs.dropna(subset=["coef"])
        c1, c2 = st.columns([2, 1])
        att_pick = c1.multiselect("Attributes", list(bsv["label"]), default=list(bsv["label"]),
                                  key="forest_attrs")
        forder = c2.radio("Order", ["Effect (high to low)", "Catalog"], horizontal=True,
                          key="forest_order")
        fb = bsv[bsv["label"].isin(att_pick)]
        if fb.empty:
            st.caption("Choose at least one attribute.")
        else:
            fo = (list(fb.sort_values("coef", ascending=False)["label"])
                  if forder.startswith("Effect") else list(fb["label"]))
            band = float(t.get("blind_spot_band", 0.03))
            st.altair_chart(CH.forest_chart(fb, band, fo), width="stretch")
            st.caption(f"An attribute whose CI lies inside the grey ±{band} band is a blind "
                       "spot: the model does not respond to it. CIs crossing the dotted lines "
                       "(mean Holm MDE) are inconclusive. Hover for p-values.")
        with st.expander("Table"):
            show(bs)
    else:
        show(bs, "Needs the main run: the pilot's 8 profiles cannot identify 11 attribute "
                 "effects.")
    rq2 = t.get("rq2") or {}
    if rq2:
        st.markdown(f"**Significant (Holm):** {', '.join(rq2['significant']) or 'none'} · "
                    f"**Blind spots:** {', '.join(rq2['blind_spots']) or 'none'}")
    sm = t.get("rq2_sameness")
    if sm:
        st.markdown("**Sameness of decisions** (share of identical action vectors, same path, "
                    "arm and month)")
        st.dataframe(pd.DataFrame(sm).T, hide_index=False)
        st.caption("If profile pairs are about as identical as C0 vs C1, the persona text "
                   "barely changes decisions.")
    st.subheader("Regression (§5.2): every term")
    reg = table(name, "analysis/regression")
    rm = table(name, "analysis/regression_models")
    me = table(name, "analysis/marginal_effects")
    rob = table(name, "analysis/robustness")
    CLUST_HELP = ("path × profile: two-way, the plan's choice (decisions share a price path and "
                  "a persona). path: one-way by price path. profile: one-way by persona, i.e. "
                  "the investor. agent run: one persona × memory arm × path, the unit that "
                  "trades; ignores correlation across paths and arms, so SEs are the smallest.")
    clusts = (list(dict.fromkeys(rm["clustering"])) if rm is not None and not rm.empty
              else ["path × profile"])
    if reg is not None and not reg.empty:
        if "status" not in reg:  # tables written before omitted terms were listed
            reg = reg.assign(status="estimated")
        st.caption(t.get("regression_model", ""))
        c1, c2 = st.columns(2)
        model = c1.radio("Model", ["LPM", "logit"], horizontal=True, key="reg_model",
                         format_func=lambda m: {"LPM": "Linear probability (plan §5.2)",
                                                "logit": "Logit (odds ratios)"}[m])
        clu = c2.selectbox("Cluster standard errors by", clusts, key="reg_clu",
                           help=CLUST_HELP)
        if rm is not None and not rm.empty:
            r = rm[(rm["model"] == model) & (rm["clustering"] == clu)]
            if not bool(r["converged"].all()):
                st.warning("The logit did not converge (likely separation): read with care.")
            cols = ["term", "coef", "se", "p", "ci_lo", "ci_hi"]
            if model == "logit":
                cols += ["odds_ratio", "or_lo", "or_hi"]
            cols += ["p_holm_dk"]
            st.dataframe(r[cols].round(4).rename(columns={
                "p_holm_dk": "p (Holm, d_k family)", "or_lo": "OR 95% low",
                "or_hi": "OR 95% high", "odds_ratio": "odds ratio"}), hide_index=True)
            st.caption(
                (f"{int(r['n_clusters_min'].iloc[0])} clusters in the smallest dimension; "
                 f"t with {int(r['n_clusters_min'].iloc[0]) - 1} df. ")
                + ("Coefficients are changes in P(sell). gain:attr_k (d_k) is the change in "
                   "P(sell | gain) − P(sell | loss), the regression counterpart of PGR − PLR."
                   if model == "LPM" else
                   "Coefficients are log-odds; odds ratio = exp(coef). For gain:attr_k, the "
                   "odds ratio is the factor by which attr_k multiplies the odds ratio of "
                   "selling a gain rather than a loss (> 1: stronger DE). Odds ratios are "
                   "not probabilities: see the marginal effects below."))
        else:
            est = reg[reg["status"] == "estimated"].drop(columns="status")
            st.dataframe(est.round(4), hide_index=True)
            st.caption(f"Logit and other clusterings: re-run analyze-study1 {name}.")
        left = reg[reg["status"] != "estimated"][["term", "status"]]
        if not left.empty:
            st.markdown(f"**Not in the fitted model ({len(left)} terms)**")
            st.dataframe(left.assign(status=left["status"].str.replace("omitted: ", "")),
                         hide_index=True)
    else:
        st.caption("Regression not estimable for this dataset (e.g. C0/C1 only).")

    def var_label(v: str) -> str:
        if v.startswith("attr_"):
            return PZ.LABELS.get(v[5:], v)
        return {"gain": "Gain (vs loss)", "mem_log": "Memory: log vs none",
                "mem_reflect": "Memory: reflect vs none", "ret": "Return this month",
                "holding_period": "Holding period", "drawdown": "Drawdown"}.get(v, v)

    if me is not None and not me.empty:
        st.subheader("Marginal effects")
        c1, c2 = st.columns([3, 2])
        kind = c1.radio("Effect", ["effect on DE", "AME on P(sell)"], horizontal=True,
                        key="me_kind", format_func=lambda k: {
                            "effect on DE": "On the DE: P(sell|gain) − P(sell|loss)",
                            "AME on P(sell)": "On P(sell)"}[k])
        me_clu = c2.selectbox("Cluster standard errors by", clusts, key="me_clu",
                              help=CLUST_HELP)
        e = me[(me["effect"] == kind) & (me["clustering"] == me_clu)].assign(
            label=lambda x: x["variable"].map(var_label))
        vlabels = list(dict.fromkeys(e["label"]))
        default = [v for v in vlabels if v not in ("Drawdown", "Return this month",
                                                   "Holding period")]
        vp = st.multiselect("Variables", vlabels, default=default, key="me_vars")
        e = e[e["label"].isin(vp)]
        if not e.empty:
            lpm_order = list(e[e["model"] == "LPM"].sort_values("estimate", ascending=False)
                             ["label"])
            xt = ("change in P(sell | gain) − P(sell | loss)" if kind == "effect on DE"
                  else "change in P(sell)")
            st.altair_chart(CH.estimates_chart(e, "model", lpm_order, xt,
                                               sorted(e["model"].unique())), width="stretch")
            wide = e.pivot(index="label", columns="model",
                           values=["estimate", "se", "p"] + (["p_holm_attr"]
                                                             if kind == "effect on DE" else []))
            wide.columns = [f"{m} {v}".replace("p_holm_attr", "p Holm") for v, m in wide.columns]
            st.dataframe(wide.reindex(lpm_order).round(4), width="stretch")
        st.caption(
            "Average over the estimation sample of counterfactual predictions, with "
            "delta-method SEs. Binary variables: 0 → 1 (memory: arm vs none); controls: dP/dx. "
            "'On the DE' is the average change in P(sell | gain) − P(sell | loss): in the LPM it "
            "equals d_k exactly; in the logit it is the marginal interaction effect on the "
            "probability scale (Ai & Norton 2003), which the log-odds coefficient is not. "
            "Holm adjusts across the 11 attributes.")

    if rob is not None and not rob.empty:
        st.subheader("Robustness to model and clustering")
        st.dataframe(rob.round(4), hide_index=True)
        if me is not None and not me.empty:
            de = me[(me["effect"] == "effect on DE") & me["variable"].str.startswith("attr_")]
            n_combo = de.groupby(["model", "clustering"]).ngroups
            sig = (de.assign(sig=de["p_holm_attr"] < 0.05)
                   .groupby("variable")["sig"].sum().astype(int)
                   .rename(f"Holm-significant in N of {n_combo} model × clustering")
                   .reset_index().assign(attribute=lambda x: x["variable"].map(var_label)))
            sig = sig.sort_values(sig.columns[1], ascending=False)
            st.markdown("**How often each attribute's effect on the DE survives Holm**")
            st.dataframe(sig[["attribute", sig.columns[1]]], hide_index=True)
            c1, c2 = st.columns([1, 3])
            rmodel = c1.radio("Model", ["LPM", "logit"], horizontal=True, key="rob_model")
            rvars = c2.multiselect("Attributes", list(dict.fromkeys(de["variable"]
                                                                    .map(var_label))),
                                   default=[], key="rob_vars",
                                   help="Empty = all attributes")
            dd = de[de["model"] == rmodel].assign(label=lambda x: x["variable"].map(var_label))
            if rvars:
                dd = dd[dd["label"].isin(rvars)]
            order = list(dd[dd["clustering"] == clusts[0]]
                         .sort_values("estimate", ascending=False)["label"])
            st.altair_chart(CH.estimates_chart(dd, "clustering", order,
                                               "effect on P(sell | gain) − P(sell | loss)",
                                               clusts), width="stretch")
            st.caption("Same point estimate under every clustering; only the 95% CI changes. "
                       "Agent-run clustering ignores that a persona's runs share a persona "
                       "and a path, so its narrow CIs are optimistic; the plan's choice is "
                       "path × profile.")

    st.subheader("Raw DE difference")
    raw = table(name, "analysis/attribute_de")
    if raw is not None and not raw.empty:
        st.dataframe(raw[["label", "raw_DE_diff", "raw_lo", "raw_hi"]].round(4)
                     .rename(columns={"raw_DE_diff": "DE level 1 − level 0",
                                      "raw_lo": "95% CI low", "raw_hi": "95% CI high"}),
                     hide_index=True)
        if t.get("regression_model") != "full §5.2 model":
            st.caption("With this few profiles, attributes that are on together cannot be "
                       "separated: descriptive only.")
    else:
        st.caption("No profiles in this dataset.")

    st.subheader("DE by variable")
    dv = table(name, "analysis/de_by_variable")
    if dv is not None and not dv.empty:
        import altair as alt
        labels = dv.drop_duplicates("variable").set_index("variable")["label"].to_dict()
        default = [v for v in ("persona_type", "memory") if v in labels]
        pick = st.multiselect("Variables", list(labels), default=default,
                              format_func=lambda v: labels[v])
        d = dv[dv["variable"].isin(pick)].copy()
        if d.empty:
            st.caption("Choose one or more variables.")
        else:
            base = alt.Chart(d).encode(
                x=alt.X("level:N", title=None, sort=alt.SortField("order")),
                tooltip=["label", "level", "runs", alt.Tooltip("PGR:Q", format=".3f"),
                         alt.Tooltip("PLR:Q", format=".3f"), alt.Tooltip("DE:Q", format=".3f"),
                         alt.Tooltip("DE_lo:Q", format=".3f"),
                         alt.Tooltip("DE_hi:Q", format=".3f")])
            bars = base.mark_bar(color="#2a78d6").encode(
                y=alt.Y("DE:Q", title="DE = PGR − PLR"))
            err = base.mark_rule(color="#52514e").encode(y="DE_lo:Q", y2="DE_hi:Q")
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#52514e").encode(y="y:Q")
            chart = (bars + err + zero).properties(width=170, height=240).facet(
                facet=alt.Facet("label:N", title=None, header=alt.Header(labelFontSize=12),
                                sort=[labels[v] for v in pick]),
                columns=4)
            st.altair_chart(chart.resolve_scale(x="independent"))
            st.caption("Pooled PGR − PLR at each level with 95% path-bootstrap intervals "
                       "(hover for PGR, PLR and run counts). Attributes and memory use the "
                       "sampled profiles only.")
    else:
        st.caption(f"Re-run analyze-study1 {name} to build this chart.")
    with st.expander("Plain-text version (regression.txt)"):
        p = an / "regression.txt"
        st.code(p.read_text() if p.exists() else "", language=None)

    st.header("Memory (RQ3)")
    if cells is not None and not cells.empty and shown:
        c1, c2 = st.columns(2)
        change = c1.radio("Plot", ["DE level", "Change vs no memory"], horizontal=True,
                          key="mem_y") == "Change vs no memory"
        bg = c2.checkbox("Show the other profiles faintly", value=False, key="mem_bg")
        mc_chart = CH.memory_chart(cells, shown, change=change, background=bg)
        if mc_chart is not None:
            st.altair_chart(mc_chart, width="stretch")
            st.caption("Personas from the sidebar. Lines that slope away from 'none' show the "
                       "record (log) or the lessons (reflect) changing the DE. Hover a line or "
                       "legend entry to highlight it.")
    else:
        image(figs / "memory_effects.png",
              "Lines that slope away from 'none' show the record or the lessons changing the DE.")
    tg = t.get("rq3_tags")
    if tg:
        image(figs / "tag_rates.png",
              "Left: how often reasoning mentions each theme by arm (past_trades should rise "
              "with memory). Right: how each attribute shifts the themes.")
        if "persona_tag_vs_n_active" in tg:
            v = tg["persona_tag_vs_n_active"]
            st.markdown(f"Persona-tag rate rises by {v['slope']:+.3f} per active attribute "
                        f"(p = {v['p']:.3f}).")
        if "reference_point_predicts_selling_winners" in tg:
            v = tg["reference_point_predicts_selling_winners"]
            st.markdown(f"Mentioning the purchase price changes the chance of selling a winner "
                        f"by {v['coef']:+.3f} (p = {v['p']:.3f}, n = {v['n']}).")
    else:
        st.caption(f"Reasoning tags not available: tag-reasoning {name}")
    agr = jload(ROOT / "results" / name / "reasoning_agreement.json")
    if agr:
        st.markdown("**Classifier against hand coding**")
        st.dataframe(pd.DataFrame(agr).T.round(3))

    st.header("Example path")
    lf = {run_label(p): p for p in log_files(name)}
    if runs is not None and not runs.empty and lf:
        c1, c2, c3 = st.columns(3)
        scen = sorted(int(s) for s in runs["scenario"].unique())
        k = c1.selectbox("Path", scen, key="ex_path")
        pers = CH.persona_order(runs["persona"].unique())
        ex_p = c2.selectbox("Persona", pers, index=pers.index("C1") if "C1" in pers else 0,
                            key="ex_persona")
        ex_arms = c3.multiselect("Memory arms", ARCHS, default=ARCHS, key="ex_arms")
        evs = {a: log_events(lf[f"{a}_{ex_p}_s{k:03d}"]) for a in ex_arms
               if f"{a}_{ex_p}_s{k:03d}" in lf}
        if not evs:
            st.caption("No logs for this choice.")
        else:
            px, tr = CH.path_frames(evs)
            stocks = list(px["stock"].unique())
            ex_s = st.multiselect("Stocks", stocks, default=stocks, key="ex_stocks")
            if ex_s:
                st.altair_chart(CH.path_chart(px, tr, ex_s))
            st.caption(f"Path {k}, {ex_p}: each arm's trades on the same prices (▼ sell, "
                       "▲ buy; arms nudged apart vertically). Hover a trade for its quantity, "
                       "average cost and that month's reasoning.")
            start = next(e for e in next(iter(evs.values())) if e["type"] == "run_start")
            with st.expander("Persona text"):
                st.write(start.get("persona_text") or "(none)")
    else:
        image(figs / "example_path.png", "Each arm's trades on the same prices.")
    st.header("Power")
    image(figs / "power_curves.png", "Power of the planned tests by profiles P and paths S.")
    with st.expander("Static figures (as saved by analyze-study1, for the paper)"):
        for f in ("de_by_persona_arm", "ratio_vs_benchmark", "forest_attributes",
                  "memory_effects", "example_path"):
            image(figs / f"{f}.png", f)


# =========================================================================== 6. findings
elif page == "Findings and conclusions":
    p = ROOT / "docs" / "prototype-2-findings.md"
    if not p.exists():
        st.info("No findings written yet (docs/prototype-2-findings.md).")
    else:
        st.caption("Written for the Luna main run (study1_v3_gpt-6-luna), whichever dataset is "
                   "selected in the sidebar. Source: docs/prototype-2-findings.md. The "
                   "Data analysis page has the interactive charts behind each number.")
        st.markdown(p.read_text(encoding="utf-8"))
