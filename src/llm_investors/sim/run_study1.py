"""Study 1: persona x memory architecture x exogenous price path.

    run-study1 --backend dummy --stage full          # rule-based agents, no API
    run-study1 --dry-run [--model gpt-6.2-sol]       # print prompts, project cost, no API
    run-study1 --stage pilot --model gpt-6-luna      # pilot, once per model (plan §2.4)
    run-study1 --stage full                          # main run with llm.model

Design v3 (plan §2.4): (P profiles + C0 + C1) x S paths x 3 arms, scenario-major. Every paid
stage first projects its cost from the exact prompts (no API) and asks before the first call;
it stops at its stage limit (budget.stage_limits) and at the overall cap (budget.cap_usd).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_investors.env import require_api_key
from llm_investors.agents.agent import Agent
from llm_investors.agents import personas as PZ
from llm_investors.envs import exogenous as X
from llm_investors.llm.cache import CacheCorrupt
from llm_investors.llm.client import CacheMiss, FatalAPIError, Halted, IncompleteLimit
from llm_investors.llm.cost import BudgetExceeded, price_usage
from llm_investors.measure import dataset, odean
from llm_investors.sim.common import (ROOT, Board, EventLog, base_args, call_stats, confirm_full_run,
                                      check_service_tier, fatal_status, finish_replay,
                                      replay_setup, report_fatal,
                                      load_config, make_llm, output_name)


class Aborted(RuntimeError):
    pass


# --------------------------------------------------------------------------- personas
def persona_roster(cfg: dict, n_profiles: int | None) -> list[dict]:
    """[{id, kind, text, profile, n_active}]. v1/v2: the fixed PERSONAS; v3: profiles + C0 + C1."""
    if cfg.get("design", "v1") != "v3":
        return [{"id": k, "kind": "fixed", "text": PZ.persona_text(k), "profile": None,
                 "n_active": None} for k in cfg["agents"]["personas"]]
    return PZ.roster(n_profiles, cfg["profiles"]["seed"])


def stage_size(cfg: dict, stage: str, args) -> tuple[int | None, int]:
    st = cfg["stages"][stage]
    prof = getattr(args, "profiles", None)
    if prof is None:                      # 0 is valid: C0 and C1 only
        prof = st.get("profiles", cfg.get("profiles", {}).get("n"))
    scen = getattr(args, "scenarios", None)
    return prof, scen if scen is not None else st["scenarios"]


def build_grid(roster: list[dict], archs: list[str], n_scen: int) -> list[tuple]:
    # Scenario-major: a budget stop removes whole matched sets (every persona x arm of the last
    # scenarios), never whole cells, so the paired design stays balanced.
    return [(p, arch, k) for k in range(n_scen) for p in roster for arch in archs]


# --------------------------------------------------------------------------- dummy agent
def dummy_policy(agent_id, t, acct, px, persona: dict, design: str):
    """Rule-based stand-in for pipeline checks. v3: attribute-dependent selling of winners,
    so the analysis has known signs to recover (loss sensitivity, realization utility and the
    C1 cue raise it; professional and market knowledge lower it)."""
    if design != "v3":
        return X.dummy_policy(agent_id, t, acct, px, persona["id"])
    prof = persona["profile"] or {}
    sell_gain = (0.25 + 0.12 * prof.get("loss_sensitivity", 0)
                 + 0.12 * prof.get("realization_utility", 0)
                 - 0.10 * prof.get("professional", 0) - 0.08 * prof.get("market_knowledge", 0)
                 + 0.25 * (persona["kind"] == "C1"))
    reply, lessons = X.dummy_policy(agent_id, t, acct, px, "experienced")
    import random
    calls = {"n": 0}

    def reply_v3() -> str:
        rng = random.Random(f"{agent_id}|{t}|{calls['n']}|v3")
        calls["n"] += 1
        if calls["n"] == 1 and rng.random() < 0.03:
            return '{"reasoning": "", "orders": []}'          # invalid on purpose
        orders = []
        for s, pos in acct.pos.items():
            ret = px[s] / pos.avg_cost - 1 if pos.shares else 0.0
            if pos.shares and ret > 0.03 and rng.random() < sell_gain:
                orders.append({"stock": s, "action": "sell", "quantity": max(1, pos.shares // 2)})
            elif pos.shares and ret < -0.03 and rng.random() < 0.12:
                orders.append({"stock": s, "action": "sell", "quantity": pos.shares})
            elif rng.random() < 0.15:
                orders.append({"stock": s, "action": "buy", "quantity": rng.randint(5, 40)})
            else:
                orders.append({"stock": s, "action": "hold", "quantity": 0})
            orders[-1] |= {"limit_price": None, "reason": "rule"}
        why = rng.choice([
            "Stock A is up since I bought it, so I lock in part of the gain.",
            "Prices move randomly, so I keep my positions and do not chase past moves.",
            "Last month I sold too early; I hold the stocks that are down and wait.",
            "As an experienced investor I rebalance toward the stocks I expect to rise."])
        return json.dumps({"reasoning": why, "orders": orders})
    return reply_v3, lessons


# --------------------------------------------------------------------------- one run
def run_one(cfg, llm, name, persona: dict, arch, k, stop: threading.Event, board: Board,
            log_root: Path | None = None):
    m, pf = cfg["market"], cfg["portfolio"]
    design = cfg.get("design", "v1")
    T = m["periods"]
    pid = persona["id"]
    run_id = f"{arch}_{pid}_s{k:03d}"
    prices, meta = X.paths_for(m, cfg["seed"], k)
    labs = X.labels(m["n_stocks"])
    log = EventLog((log_root or ROOT / "logs") / name / f"run_{run_id}.jsonl")
    log("run_start", run_id=run_id, study="study1", design=design, architecture=arch,
        persona=pid, persona_kind=persona["kind"], profile=persona["profile"],
        n_active=persona["n_active"], persona_text=persona["text"],
        scenario=k, scenario_meta=meta, prices=prices.tolist(), labels=labs,
        model=cfg["llm"]["model"], backend=cfg["llm"]["backend"])
    dummy = cfg["llm"]["backend"] == "dummy" or llm.dry_run
    agent = Agent(agent_id=run_id, persona=pid, architecture=arch,
                  env=X.env_spec(m["n_stocks"], design), llm=llm, log=log,
                  json_retries=cfg["agents"]["json_retries"],
                  reflect_after=tuple(cfg["agents"]["reflect_after"]),
                  replicate_id=cfg["llm"].get("replicate_id"),
                  show_pnl=m.get("show_pnl", True),
                  persona_prompt=persona["text"] if design == "v3" else None)
    acct = X.Account(pf["initial_cash"], {s: X.Position(pf["initial_shares"], m["start_price"], 0)
                                         for s in labs})
    peak = acct.value(dict(zip(labs, prices[0])))
    prev_value = None
    status, fatal = "complete", None
    hist = m.get("price_history", 0)
    try:
        for t in range(1, T + 1):
            if stop.is_set():
                raise Aborted("budget cap reached in another run")
            px = {s: float(prices[t, i]) for i, s in enumerate(labs)}
            value = acct.value(px)
            peak = max(peak, value)
            drawdown = value / peak - 1
            before = {s: (p.shares, p.avg_cost, p.opened) for s, p in acct.pos.items() if p.shares}
            reply, lessons = dummy_policy(run_id, t, acct, px, persona, design) if dummy \
                else (None, None)
            recent = X.recent_prices(prices, t, labs, hist) if isinstance(hist, int) and hist \
                else None
            history = X.price_history(prices, t, labs) if hist == "all" else None
            orders = agent.decide(t, X.state_block(t, T, acct, px, m.get("show_pnl", True),
                                                   recent, history), dummy=reply)
            trades, rejected = acct.execute(orders, px, t)
            for r in rejected:
                log("order_rejected", agent=run_id, period=t, **r)
            for tr in trades:
                log("trade", agent=run_id, period=t, **tr)
            sold = {tr["stock"]: tr["quantity"] for tr in trades if tr["action"] == "sell"}
            obs = [{"period": t, "agent": run_id, "stock": s, "shares": sh,
                    "avg_cost": round(c, 6), "price": px[s], "ret": px[s] / c - 1,
                    "gain": None if px[s] == c else int(px[s] > c),
                    "sold": s in sold, "sell": int(s in sold), "qty_sold": sold.get(s, 0),
                    "holding_period": t - op, "drawdown": drawdown,
                    "sale_day": int(bool(sold))}
                   for s, (sh, c, op) in before.items()]
            log("period_end", period=t, cash=acct.cash, value=acct.value(px),
                holdings={s: {"shares": p.shares, "avg_cost": p.avg_cost}
                          for s, p in acct.pos.items()},
                orders=orders, observations=obs, reasoning=agent.last_reasoning)
            agent.end_period(t, {"period": t, "value": value, "prev_value": prev_value,
                                 "trades": trades}, T, dummy_lessons=lessons)
            prev_value = value
            acts = [f"{o['stock']} {o['action']} {o['quantity']}" for o in orders
                    if o["action"] != "hold"]
            why = agent.last_reasoning or next((o["reason"] for o in orders
                                                if o["action"] != "hold"), orders[0]["reason"])
            board.update(run_id, period=f"{t}/{T}",
                         prices=" ".join(f"{s}{px[s]:.0f}" for s in labs),
                         decisions=", ".join(acts) or "hold all", reason=why[:80])
    except CacheCorrupt as e:
        # A broken shared-cache file must never turn into a silent paid call. Stop everything.
        stop.set()
        status = "cache_corrupt"
        log("cache_corrupt", message=str(e))
        board.echo(f"[FATAL] {e}")
    except CacheMiss:
        # Offline replay: the prompts differ from the cached run. Stop everything ($0).
        stop.set()
        status = "cache_miss"
    except FatalAPIError as e:
        # Quota/billing/non-retryable error: stop every run now (logged by the agent).
        stop.set()
        status, fatal = fatal_status(e), e.to_dict()
        board.echo(f"[FATAL] {run_id}: {e} -> stopping all runs")
    except BudgetExceeded as e:
        stop.set()
        status = "budget_stop"
        log("budget_stop", message=str(e))
    except IncompleteLimit:
        status = "incomplete_stop"   # only this run stops; logged by the agent, never a hold
    except (Aborted, Halted):
        status = "aborted"
    final = acct.value({s: float(prices[T, i]) for i, s in enumerate(labs)})
    log("run_end", status=status, final_value=final, json_invalid=agent.n_invalid,
        incomplete=agent.n_incomplete,
        fallbacks=agent.n_fallback, lessons=agent.memory.lessons)
    log.close()
    board.finish(run_id)
    return {"run_id": run_id, "status": status, "fatal": fatal}


# --------------------------------------------------------------------------- projection
class _NullBoard:
    def update(self, *a, **k): pass
    def finish(self, *a, **k): pass
    def echo(self, *a, **k): pass


def estimate(cfg: dict, roster: list[dict], n_scen: int) -> dict:
    """Project a stage's cost from its exact prompts, with no API call: scenario 0 is simulated
    for every persona x arm with dry-run replies, and its prompt sizes are scaled to n_scen.
    Returns calls, input tokens (chars/4) and typical / upper-bound USD."""
    c = {**cfg, "llm": {**cfg["llm"]}}
    tmp = Path(tempfile.mkdtemp(prefix="study1_estimate_"))
    try:
        llm = make_llm(c, "estimate", True, 0, echo=lambda *a, **k: None)
        llm.cache.root = tmp / "cache"          # never touch the real cache
        stop = threading.Event()
        for p in roster:
            for arch in cfg["agents"]["architectures"]:
                run_one(c, llm, "estimate", p, arch, 0, stop, _NullBoard(), log_root=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pr = llm.projection
    calls = pr["calls"] * n_scen
    inp = pr["input_tokens"] * n_scen
    prices = cfg["llm"]["price_per_mtok"]
    typ_out = cfg["llm"].get("est_output_tokens", cfg["llm"]["max_output_tokens"])
    typical = price_usage(prices, {"input_tokens": inp, "output_tokens": calls * typ_out})
    upper = price_usage(prices, {"input_tokens": inp,
                                 "output_tokens": calls * cfg["llm"]["max_output_tokens"]})
    runs = len(roster) * len(cfg["agents"]["architectures"]) * n_scen
    return {"runs": runs, "calls": calls, "input_tokens": inp,
            "mean_input_tokens": inp / max(1, calls), "typical_usd": typical,
            "upper_usd": upper, "typical_output_tokens": typ_out,
            "per_run_typical_usd": typical / max(1, runs)}


def print_estimate(e: dict, limit: float | None, cap_left: float | None, label: str):
    print(f"\nProjection for {label}: {e['runs']:,} runs, {e['calls']:,} calls, "
          f"≈ {e['mean_input_tokens']:.0f} input tokens per call (chars/4).")
    print(f"  typical (≈{e['typical_output_tokens']} output tokens/call): ${e['typical_usd']:.2f}"
          f"  (${e['per_run_typical_usd']:.4f} per run)")
    print(f"  upper bound (every call at max_output_tokens):      ${e['upper_usd']:.2f}")
    if limit is not None:
        print(f"  stage limit ${limit:.2f}" + (f"; overall budget left ${cap_left:.2f}"
                                               if cap_left is not None else ""))


def max_affordable_scenarios(e: dict, n_scen: int, limit: float) -> int:
    per_scen = e["typical_usd"] / max(1, n_scen)
    return int(limit // per_scen) if per_scen > 0 else n_scen


# --------------------------------------------------------------------------- main
def apply_model(cfg: dict, model: str | None, backend: str, paid: bool) -> str:
    """Set llm.model, prices and tier for `--model`; return the model id."""
    if not model:
        return cfg["llm"]["model"]
    prices = (cfg["llm"].get("model_prices") or {}).get(model)
    if prices is None and model != cfg["llm"]["model"] and backend == "openai" and paid:
        sys.exit(f"No prices for {model!r}. Add llm.model_prices.{model} (USD per 1M tokens: "
                 "input, cached_input, output) to config/study1.yaml so the budget cap is "
                 "enforced correctly.")
    cfg["llm"]["model"] = model
    if prices is not None:
        prices = dict(prices)
        if "service_tier" in prices:
            cfg["llm"]["service_tier"] = prices.pop("service_tier")
        cfg["llm"]["price_per_mtok"] = prices
    return model


def variant_name(cfg: dict, model_variant: str) -> str:
    design = cfg.get("design", "v1")
    return "_".join(v for v in ("" if design == "v1" else design,
                                "" if cfg["market"].get("show_pnl", True) else "nopnl",
                                model_variant) if v)


def stage_limit(cfg: dict, stage: str, model: str) -> float | None:
    lim = cfg["budget"].get("stage_limits") or {}
    return lim.get(f"pilot_{model}") if stage == "pilot" else lim.get("full")


def main():
    """Run, then send a phone notification (ntfy, NTFY_TOPIC in .env) when a real run ends,
    whether it completed, stopped on an error or was interrupted."""
    from llm_investors.notify import notify
    info: dict = {}
    try:
        _main(info)
    except BaseException as e:  # noqa: BLE001
        if info.get("started"):
            code = e.code if isinstance(e, SystemExit) else type(e).__name__
            if not (isinstance(e, SystemExit) and e.code in (0, None)):
                notify(f"Study 1 {info['name']} STOPPED", f"{info.get('summary', '')}\n"
                       f"exit: {code}. See logs/{info['name']}/.", tags="warning",
                       priority="high")
            else:
                notify(f"Study 1 {info['name']} finished", info.get("summary", ""),
                       tags="white_check_mark")
        raise
    if info.get("started"):
        notify(f"Study 1 {info['name']} finished", info.get("summary", ""),
               tags="white_check_mark")


def _main(info: dict):
    ap = base_args(__doc__)
    ap.add_argument("--scenarios", type=int, help="override the stage's number of paths (S)")
    ap.add_argument("--profiles", type=int, help="override the stage's number of profiles (P)")
    ap.add_argument("--no-pnl", action="store_true",
                    help="ablation: hide the gain/loss-vs-cost column and the %% vs cost in the "
                         "log history (market.show_pnl: false); outputs go to study1_nopnl*")
    ap.add_argument("--model", help="override llm.model (e.g. gpt-6.2-sol); prices come from "
                    "llm.model_prices. v3 outputs always go to study1_v3_<model>*")
    args = ap.parse_args()
    cfg = load_config("study1")
    if args.backend:
        cfg["llm"]["backend"] = args.backend
    backend = cfg["llm"]["backend"]
    if args.no_pnl:
        cfg["market"]["show_pnl"] = False
    design = cfg.get("design", "v1")
    paid = backend == "openai" and not (args.dry_run or args.replay)
    model = apply_model(cfg, args.model, backend, paid)
    # v3 always names the model (two pilots, one per model); v2 only when overridden
    model_variant = model if design == "v3" else (
        args.model if args.model and args.model != load_config("study1")["llm"]["model"] else "")
    variant = variant_name(cfg, model_variant)
    name = output_name("study1", backend, args.dry_run, args.stage, variant)
    n_prof, n_scen = stage_size(cfg, args.stage, args)
    roster = persona_roster(cfg, n_prof)
    archs = cfg["agents"]["architectures"]
    proj_file = ROOT / "results" / output_name("study1", backend, False, "pilot", variant) \
        / "pilot_projection.json"
    log_name = replay_setup(args, backend, name)
    limit = stage_limit(cfg, args.stage, model)

    if paid:
        require_api_key()      # from .env (git-ignored) or the environment; fails fast if missing
        est = estimate(cfg, roster, n_scen)
        print_estimate(est, limit, None, f"{name} ({args.stage}, {model})")
        if limit is not None and est["typical_usd"] > limit:
            k = max_affordable_scenarios(est, n_scen, limit)
            per_run = est["per_run_typical_usd"]
            hint = (f"e.g. --scenarios {k}" if k >= 1 else
                    f"even one path ({len(roster) * len(archs)} runs, "
                    f"≈ ${per_run * len(roster) * len(archs):.2f}) is over the limit: the limit "
                    f"buys about {int(limit // per_run)} runs at ${per_run:.4f} each, so also "
                    f"lower --profiles (0 = C0 and C1 only) or raise "
                    f"budget.stage_limits in config/study1.yaml")
            sys.exit(f"The typical projection exceeds the stage limit ${limit:.2f}. Shrink the "
                     f"stage first (plan §8): {hint}.")
        if args.stage == "full" and not confirm_full_run(cfg, args, proj_file):
            return
        if args.stage == "pilot" and not args.yes and \
                input("Start the paid pilot? [y/N] ").strip().lower() != "y":
            return

    for old in [*(ROOT / "logs" / log_name).glob("run_*.jsonl"),
                *(ROOT / "logs" / log_name).glob("FATAL_ERROR.json")]:
        old.unlink()  # each invocation rewrites the stage's logs (replays are free)
    grid = build_grid(roster, archs, n_scen)
    info.update(name=name, started=not args.dry_run and backend == "openai",
                summary=f"{args.stage}, {model}: {len(grid)} runs started")
    if args.dry_run:
        grid = grid[: len(roster) * len(archs)]  # one scenario each
    stop = threading.Event()
    board = Board(f"Study 1 [{name}, {args.stage}]", ["period", "prices", "decisions", "reason"],
                  None, enabled=not args.no_live and not args.dry_run)
    llm = make_llm(cfg, name, args.dry_run, args.show_prompts, echo=board.echo,
                   offline=args.replay, retry_incomplete=args.retry_incomplete,
                   stage_limit=limit)
    board.cost, board.total = llm.cost, len(grid)
    with board, ThreadPoolExecutor(max_workers=args.workers or cfg["run"]["workers"]) as ex:
        results = list(ex.map(lambda g: run_one(cfg, llm, log_name, g[0], g[1], g[2], stop,
                                                board), grid))

    statuses = {s: sum(r["status"] == s for r in results) for s in {r["status"] for r in results}}
    print(f"\nruns: {statuses}")
    print(f"cost: {json.dumps(llm.cost.summary())}")
    cs = llm.cost.summary()
    info["summary"] = (f"{args.stage}, {model}: runs {statuses}; spent ${cs['spent_usd']:.3f} "
                       f"this stage, ${cs['spent_total_usd']:.3f} of ${cs['cap_usd']:.2f} total; "
                       f"{cs['api_calls']} API calls, {cs['cache_hits']} cache hits")
    if args.dry_run:
        report_dry_run(cfg, roster, n_scen, args.stage, model)
        return
    if args.replay and (code := finish_replay(results, name, log_name)):
        sys.exit(code)
    dfs = dataset.build(name)
    out = ROOT / "results" / name
    for k, df in dfs.items():
        dataset._save(df, out, k)
    summarize(dfs["runs"])
    if backend == "openai" and not args.replay and llm.cost.session["api_calls"]:
        from llm_investors.sim.cache_stats import reminder
        if (msg := reminder("study1")):
            print("\n" + msg)
    code = report_fatal(results, ROOT / "logs" / name)
    if backend == "openai" and not args.replay and not args.dry_run:
        # logs/ is git-ignored; a gzipped copy in published/ lets a deployed viewer show them
        from llm_investors.sim.publish_logs import publish, summary_line
        print("\n" + summary_line(publish(log_name)))
    if code:
        sys.exit(code)        # no cost projection from an interrupted pilot
    if args.stage == "pilot" and backend == "openai" and not args.replay:
        line = project(cfg, name, llm, proj_file, roster, n_scen)
        if line:
            info["summary"] += "\n" + line


def summarize(runs):
    if runs.empty:
        return
    by = "persona_kind" if "persona_kind" in runs and runs["persona_kind"].notna().any() \
        else "persona"
    print(f"\nPooled PGR / PLR / DE by architecture x {by} (sale days only):")
    for (arch, pers), g in runs.groupby(["architecture", by]):
        p = odean.ratios(g[["RG", "RL", "PG", "PL"]].sum())
        f = lambda x: " n/a " if x is None else f"{x:+.3f}"
        print(f"  {arch:<8} {pers:<13} PGR {f(p['PGR'])}  PLR {f(p['PLR'])}  DE {f(p['DE'])}"
              f"   invalid {g['invalid_replies'].sum():>3}  fallbacks {g['fallbacks'].sum():>3}"
              f"  rejected orders {g['orders_rejected'].sum():>3}"
              f"  incomplete {g['incomplete'].sum():>3}"
              f"  incomplete stops {g['incomplete_stops'].sum():>2}")


def project(cfg, name, llm, proj_file, pilot_roster, pilot_scen):
    """After a pilot: measured cost per run and the projected main-run cost (plan §2.5)."""
    st = call_stats(ROOT / "logs" / name, llm.cost.price)
    if not st["calls"]:
        return
    check_service_tier(st, cfg["llm"].get("service_tier"))
    archs = cfg["agents"]["architectures"]
    pilot_runs = len(pilot_roster) * len(archs) * pilot_scen
    cost_per_run = st["mean_cost_usd"] * st["calls"] / pilot_runs
    calls_per_run = st["calls"] / pilot_runs
    full_prof, full_scen = cfg["stages"]["full"].get("profiles"), cfg["stages"]["full"]["scenarios"]
    full_runs = ((full_prof + 2) if cfg.get("design") == "v3" else
                 len(cfg["agents"]["personas"])) * len(archs) * full_scen
    full_cost = full_runs * cost_per_run
    frac_new = 1 - pilot_runs / full_runs
    p = {**st, "pilot_runs": pilot_runs, "calls_per_run": calls_per_run,
         "cost_per_run_usd": cost_per_run, "model": cfg["llm"]["model"],
         "projected_full_runs": full_runs, "projected_full_calls": round(full_runs * calls_per_run),
         "projected_full_cost_usd": full_cost,
         "projected_remaining_cost_usd": full_cost * frac_new, "cap_usd": cfg["budget"]["cap_usd"],
         "stage_limit_full_usd": (cfg["budget"].get("stage_limits") or {}).get("full")}
    proj_file.parent.mkdir(parents=True, exist_ok=True)
    proj_file.write_text(json.dumps(p, indent=2))
    print(f"\nMeasured per call: {st['mean_input_tokens']:.0f} input, "
          f"{st['mean_output_tokens']:.0f} output ({st['mean_reasoning_tokens']:.0f} reasoning) "
          f"tokens, ${st['mean_cost_usd']:.6f}; ${cost_per_run:.4f} per run")
    proj_line = (f"Projected full run ({full_runs:,} runs): ≈ ${full_cost:.2f} "
                 f"(${cost_per_run:.4f}/run)")
    print(f"Projected full run ({full_runs:,} runs): ≈ ${full_cost:.2f} "
          f"(≈ ${full_cost * frac_new:.2f} new, if the same model) vs stage limit "
          f"${p['stage_limit_full_usd']}.")
    print(f"Next (free): build-dataset {name} && power-study1 {name}")
    return proj_line


def report_dry_run(cfg, roster, n_scen, stage, model):
    est = estimate(cfg, roster, n_scen)
    print_estimate(est, stage_limit(cfg, stage, model), None, f"{stage} stage, {model}")
    print("(One scenario per persona x architecture was simulated with dummy replies; the "
          "prompts above are exact.)")


if __name__ == "__main__":
    main()
