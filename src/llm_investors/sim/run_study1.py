"""Study 1: memory architecture x persona x exogenous price scenario.

    run-study1 --backend dummy --stage full   # step 1: rule-based agents, no API
    run-study1 --dry-run                      # print prompts, project cost
    run-study1 --stage pilot                  # step 2: 5 scenarios x 3 x 3
    run-study1 --stage full                   # step 3: asks for confirmation
"""
from __future__ import annotations

import itertools
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from llm_investors.env import require_api_key
from llm_investors.agents.agent import Agent
from llm_investors.envs import exogenous as X
from llm_investors.llm.cache import CacheCorrupt
from llm_investors.llm.client import CacheMiss, FatalAPIError, Halted, IncompleteLimit
from llm_investors.llm.cost import BudgetExceeded
from llm_investors.measure import dataset, odean
from llm_investors.sim.common import (ROOT, Board, EventLog, base_args, call_stats, confirm_full_run,
                                      check_service_tier, fatal_status, finish_replay,
                                      replay_setup, report_fatal,
                        load_config, make_llm, output_name)


class Aborted(RuntimeError):
    pass


def run_one(cfg, llm, name, persona, arch, k, stop: threading.Event, board: Board):
    m, pf = cfg["market"], cfg["portfolio"]
    T = m["periods"]
    run_id = f"{arch}_{persona}_s{k:03d}"
    prices, meta = X.scenario_paths(m, cfg["seed"], k)
    labs = X.labels(m["n_stocks"])
    log = EventLog(ROOT / "logs" / name / f"run_{run_id}.jsonl")
    log("run_start", run_id=run_id, study="study1", architecture=arch, persona=persona,
        scenario=k, scenario_meta=meta, prices=prices.tolist(), labels=labs,
        model=cfg["llm"]["model"], backend=cfg["llm"]["backend"])
    dummy = cfg["llm"]["backend"] == "dummy" or llm.dry_run
    agent = Agent(agent_id=run_id, persona=persona, architecture=arch,
                  env=X.env_spec(m["n_stocks"], cfg.get("design", "v1")), llm=llm, log=log,
                  json_retries=cfg["agents"]["json_retries"],
                  reflect_after=tuple(cfg["agents"]["reflect_after"]),
                  replicate_id=cfg["llm"].get("replicate_id"),
                  show_pnl=m.get("show_pnl", True))
    acct = X.Account(pf["initial_cash"], {s: X.Position(pf["initial_shares"], m["start_price"], 0)
                                         for s in labs})
    peak = acct.value(dict(zip(labs, prices[0])))
    prev_value = None
    status, fatal = "complete", None
    try:
        for t in range(1, T + 1):
            if stop.is_set():
                raise Aborted("budget cap reached in another run")
            px = {s: float(prices[t, i]) for i, s in enumerate(labs)}
            value = acct.value(px)
            peak = max(peak, value)
            drawdown = value / peak - 1
            before = {s: (p.shares, p.avg_cost, p.opened) for s, p in acct.pos.items() if p.shares}
            reply, lessons = X.dummy_policy(run_id, t, acct, px, persona) if dummy else (None, None)
            recent = X.recent_prices(prices, t, labs, m.get("price_history", 0)) \
                if m.get("price_history", 0) else None
            orders = agent.decide(t, X.state_block(t, T, acct, px, m.get("show_pnl", True),
                                                   recent), dummy=reply)
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
                orders=orders, observations=obs)
            agent.end_period(t, {"period": t, "value": value, "prev_value": prev_value,
                                 "trades": trades}, T, dummy_lessons=lessons)
            prev_value = value
            acts = [f"{o['stock']} {o['action']} {o['quantity']}" for o in orders
                    if o["action"] != "hold"]
            why = next((o["reason"] for o in orders if o["action"] != "hold"), orders[0]["reason"])
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


def main():
    ap = base_args(__doc__)
    ap.add_argument("--scenarios", type=int, help="override the stage's scenario count")
    ap.add_argument("--no-pnl", action="store_true",
                    help="ablation: hide the gain/loss-vs-cost column and the %% vs cost in the "
                         "log history (market.show_pnl: false); outputs go to study1_nopnl*")
    args = ap.parse_args()
    cfg = load_config("study1")
    if args.backend:
        cfg["llm"]["backend"] = args.backend
    backend = cfg["llm"]["backend"]
    if args.no_pnl:
        cfg["market"]["show_pnl"] = False
    design = cfg.get("design", "v1")
    variant = "_".join(v for v in ("" if design == "v1" else design,
                                   "" if cfg["market"].get("show_pnl", True) else "nopnl") if v)
    name = output_name("study1", backend, args.dry_run, args.stage, variant)
    n_scen = args.scenarios or cfg["stages"][args.stage]["scenarios"]
    proj_file = ROOT / "results" / output_name("study1", backend, False, "pilot", variant) \
        / "pilot_projection.json"
    log_name = replay_setup(args, backend, name)
    if backend == "openai" and not (args.dry_run or args.replay):
        require_api_key()      # from .env (git-ignored) or the environment; fails fast if missing
    if args.stage == "full" and not confirm_full_run(cfg, args, proj_file):
        return

    for old in [*(ROOT / "logs" / log_name).glob("run_*.jsonl"),
                *(ROOT / "logs" / log_name).glob("FATAL_ERROR.json")]:
        old.unlink()  # each invocation rewrites the stage's logs (replays are free)
    a = cfg["agents"]
    # Scenario-major: a budget stop removes whole matched sets (all personas x arms of the last
    # scenarios), never whole cells, so the paired design stays balanced.
    grid = [(p, arch, k) for k in range(n_scen) for p in a["personas"]
            for arch in a["architectures"]]
    if args.dry_run:
        grid = grid[: len(a["personas"]) * len(a["architectures"])]  # one scenario each
    stop = threading.Event()
    board = Board(f"Study 1 [{name}, {args.stage}]", ["period", "prices", "decisions", "reason"],
                  None, enabled=not args.no_live and not args.dry_run)
    llm = make_llm(cfg, name, args.dry_run, args.show_prompts, echo=board.echo,
                   offline=args.replay, retry_incomplete=args.retry_incomplete)
    board.cost, board.total = llm.cost, len(grid)
    with board, ThreadPoolExecutor(max_workers=args.workers or cfg["run"]["workers"]) as ex:
        results = list(ex.map(lambda g: run_one(cfg, llm, log_name, g[0], g[1], g[2], stop,
                                                board), grid))

    statuses = {s: sum(r["status"] == s for r in results) for s in {r["status"] for r in results}}
    print(f"\nruns: {statuses}")
    print(f"cost: {json.dumps(llm.cost.summary())}")
    if args.dry_run:
        report_dry_run(cfg, llm, n_scen)
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
    if code:
        sys.exit(code)        # no cost projection from an interrupted pilot
    if args.stage == "pilot" and backend == "openai" and not args.replay:
        project(cfg, name, llm, proj_file)


def summarize(runs):
    if runs.empty:
        return
    print("\nPooled PGR / PLR / DE by architecture x persona (sale days only):")
    for (arch, pers), g in runs.groupby(["architecture", "persona"]):
        p = odean.ratios(g[["RG", "RL", "PG", "PL"]].sum())
        f = lambda x: " n/a " if x is None else f"{x:+.3f}"
        print(f"  {arch:<8} {pers:<13} PGR {f(p['PGR'])}  PLR {f(p['PLR'])}  DE {f(p['DE'])}"
              f"   invalid {g['invalid_replies'].sum():>3}  fallbacks {g['fallbacks'].sum():>3}"
              f"  rejected orders {g['orders_rejected'].sum():>3}"
              f"  incomplete {g['incomplete'].sum():>3}"
              f"  incomplete stops {g['incomplete_stops'].sum():>2}")


def full_calls(cfg) -> int:
    a, T = cfg["agents"], cfg["market"]["periods"]
    n = cfg["stages"]["full"]["scenarios"] * len(a["personas"])
    refl = len([r for r in a["reflect_after"] if r < T])
    return n * len(a["architectures"]) * T + n * refl * ("reflect" in a["architectures"])


def project(cfg, name, llm, proj_file):
    st = call_stats(ROOT / "logs" / name, llm.cost.price)
    if not st["calls"]:
        return
    check_service_tier(st, cfg["llm"].get("service_tier"))
    calls = full_calls(cfg) * (1 + st["json_retry_rate"])
    full_cost = calls * st["mean_cost_usd"]
    frac_new = 1 - cfg["stages"]["pilot"]["scenarios"] / cfg["stages"]["full"]["scenarios"]
    p = {**st, "projected_full_calls": round(calls), "projected_full_cost_usd": full_cost,
         "projected_remaining_cost_usd": full_cost * frac_new, "cap_usd": cfg["budget"]["cap_usd"]}
    proj_file.parent.mkdir(parents=True, exist_ok=True)
    proj_file.write_text(json.dumps(p, indent=2))
    print(f"\nMeasured per call: {st['mean_input_tokens']:.0f} input, "
          f"{st['mean_output_tokens']:.0f} output ({st['mean_reasoning_tokens']:.0f} reasoning) "
          f"tokens, ${st['mean_cost_usd']:.6f}")
    print(f"Projected full run: {calls:,.0f} calls ≈ ${full_cost:.2f} "
          f"(≈ ${full_cost * frac_new:.2f} new) vs cap ${cfg['budget']['cap_usd']:.2f}.")
    print("Stop here and confirm before: run-study1 --stage full")


def report_dry_run(cfg, llm, n_scen):
    p = llm.projection
    per = llm.cost.price({"input_tokens": p["input_tokens"] / max(1, p["calls"]),
                          "output_tokens": cfg["llm"]["max_output_tokens"]})
    n = full_calls(cfg)
    print(f"\nDry run: {p['calls']} prompts built (one scenario per persona x architecture), "
          f"≈ {p['input_tokens'] / max(1, p['calls']):.0f} input tokens each (chars/4).")
    print(f"Upper-bound projection, full stage ({n:,} calls, every call at max_output_tokens): "
          f"≈ ${n * per:.2f}; cap ${cfg['budget']['cap_usd']:.2f}.")


if __name__ == "__main__":
    main()
