"""Study 2: LLM agents trade one stock with each other in a call-auction market.

    run-study2 --backend dummy --stage full   # step 4: rule-based agents, no API
    run-study2 --dry-run                      # print prompts, project cost
    run-study2 --stage pilot                  # step 5a: ~10 agents x 10 periods
    run-study2 --stage full                   # step 5b: ~20 market runs
"""
from __future__ import annotations

import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from llm_investors.env import require_api_key
from llm_investors.agents.agent import Agent
from llm_investors.envs import market as M
from llm_investors.llm.cache import CacheCorrupt
from llm_investors.llm.client import CacheMiss, FatalAPIError, Halted, IncompleteLimit
from llm_investors.llm.cost import BudgetExceeded
from llm_investors.measure import dataset
from llm_investors.sim.common import (ROOT, Board, EventLog, base_args, call_stats, confirm_full_run,
                                      check_service_tier, fatal_status, finish_replay,
                                      replay_setup, report_fatal,
                        load_config, make_llm, output_name)


class Aborted(RuntimeError):
    pass


def cell_params(cfg: dict, cell: dict) -> dict:
    p = {**cfg["baseline"], "periods": cfg["market"]["periods"], **cell}
    p.pop("runs", None)
    return p


def persona_list(mix: dict, n: int, rng: random.Random) -> list[str]:
    """Exact persona counts from shares (largest remainder), in a shuffled order."""
    raw = {k: v * n / sum(mix.values()) for k, v in mix.items()}
    cnt = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - cnt[k]))[: n - sum(cnt.values())]:
        cnt[k] += 1
    out = [k for k, c in cnt.items() for _ in range(c)]
    rng.shuffle(out)
    return out


def run_market(cfg, llm, name, cell_name, P, rep, stop: threading.Event, board: Board,
               agent_workers: int):
    m = cfg["market"]
    T, n, divs = P["periods"], P["n_agents"], m["dividends"]
    ev_div = sum(divs) / len(divs)
    run_id = f"{cell_name}_r{rep:02d}"
    # Common random numbers: replicate r draws the same dividends and endowments in every cell.
    rng = random.Random(cfg["seed"] + rep)
    personas = persona_list(cfg["agent_mixes"][P["agent_mix"]], n, rng)
    fv1 = M.fundamental_value(1, T, divs)
    shares = m["endowment_shares"]
    endow = [(m["wealth"] - shares[i % len(shares)] * fv1, shares[i % len(shares)])
             for i in range(n)]
    assert all(c >= 0 for c, _ in endow), "wealth too small for the share endowments"
    rng.shuffle(endow)
    dividends = [rng.choice(divs) for _ in range(T)]
    traders = {f"t{i:02d}": M.Trader(float(c), int(s), fv1, 0) for i, (c, s) in enumerate(endow)}
    ids = list(traders)
    design = cfg.get("design", "v1")
    two_sided = design != "v1"
    spec = M.env_spec(T, divs, P["price_limit"], P["transaction_tax"], design)
    dummy = cfg["llm"]["backend"] == "dummy" or llm.dry_run
    log = EventLog(ROOT / "logs" / name / f"run_{run_id}.jsonl")
    total_shares = sum(tr.shares for tr in traders.values())
    log("run_start", run_id=run_id, study="study2", cell=cell_name, replicate=rep,
        agent_mix=P["agent_mix"], memory=P["memory"], architecture=P["memory"],
        price_limit=P["price_limit"], transaction_tax=P["transaction_tax"], n_agents=n,
        design=design, two_sided=two_sided,
        periods=T, dividends_drawn=dividends, total_shares=total_shares,
        agents=[{"agent": a, "persona": p, "cash": traders[a].cash, "shares": traders[a].shares}
                for a, p in zip(ids, personas)],
        model=cfg["llm"]["model"], backend=cfg["llm"]["backend"])
    base_rep = cfg["llm"].get("replicate_id")
    agents = {a: Agent(agent_id=a, persona=p, architecture=P["memory"], env=spec, llm=llm,
                       log=log, json_retries=cfg["agents"]["json_retries"],
                       reflect_after=tuple(cfg["agents"]["reflect_after"]),
                       # Stable per (replicate, trader): keeps agents and market runs distinct
                       # without breaking cache replay.
                       replicate_id=f"{rep}-{i}" + (f"-{base_rep}" if base_rep is not None else ""))
              for i, (a, p) in enumerate(zip(ids, personas))}
    history, last_price, status, fatal = [], fv1, "complete", None
    prev_value = {a: None for a in ids}
    pool = ThreadPoolExecutor(max_workers=agent_workers)
    try:
        for t in range(1, T + 1):
            if stop.is_set():
                raise Aborted("budget cap reached in another run")
            fv = M.fundamental_value(t, T, divs)

            def decide(a):
                tr = traders[a]
                reply, _ = (M.dummy_policy(f"{rep}/{a}", t, tr, fv, last_price, two_sided)
                            if dummy else (None, None))
                return agents[a].decide(t, M.state_block(t, T, tr, history, ev_div, last_price),
                                        dummy=reply,
                                        check=M.quote_check(tr) if two_sided else None)
            # orders[a] is a list: [buy, sell] (design v2), [one order] (v1) or [hold] (fallback)
            orders = dict(zip(ids, pool.map(decide, ids)))

            before = {a: (tr.shares, tr.avg_cost, tr.opened) for a, tr in traders.items()}
            cash0 = sum(tr.cash for tr in traders.values())
            buys, sells = [], []
            for a in ids:
                checked = []
                for o in orders[a]:
                    o, rej = M.validate(o, traders[a])
                    if rej:
                        log("order_rejected", agent=a, period=t, **rej)
                    checked.append(o)
                    if o["action"] in ("buy", "sell") and o["quantity"] > 0:
                        (buys if o["action"] == "buy" else sells).append(
                            {"id": a, "quantity": o["quantity"], "limit_price": o["limit_price"]})
                orders[a] = checked
            res = M.clear_call_auction(buys, sells, last_price, P["price_limit"],
                                       m.get("min_imbalance_tiebreak", True))
            # buy limit < sell limit, so no agent can be on both sides of one clearing price
            assert not ({b["id"] for b in buys if b["limit_price"] >= res["price"] - 1e-9}
                        & {o["id"] for o in sells if o["limit_price"] <= res["price"] + 1e-9}) \
                or not res["traded"]
            p, tax_paid, trades = res["price"], 0.0, {a: [] for a in ids}
            for a, q in res["fills"].items():
                tr = traders[a]
                if q > 0:
                    tr.avg_cost = (tr.shares * tr.avg_cost + q * p) / (tr.shares + q) if tr.shares else p
                    if tr.shares == 0:
                        tr.opened = t
                    tr.shares += q
                    tr.cash -= q * p
                    trades[a].append({"stock": "A", "action": "buy", "quantity": q, "price": p,
                                      "avg_cost": before[a][1]})
                else:
                    tax = -q * p * P["transaction_tax"]
                    tr.shares += q
                    tr.cash += -q * p - tax
                    tax_paid += tax
                    trades[a].append({"stock": "A", "action": "sell", "quantity": -q, "price": p,
                                      "avg_cost": before[a][1], "tax": tax})
                log("trade", agent=a, period=t, side="buy" if q > 0 else "sell",
                    quantity=abs(q), price=p)
            d = dividends[t - 1]
            for tr in traders.values():
                tr.cash += d * tr.shares
            # Conservation: shares are fixed; cash changes only by dividends and tax.
            assert sum(tr.shares for tr in traders.values()) == total_shares
            cash1 = sum(tr.cash for tr in traders.values())
            assert abs(cash1 - (cash0 + d * total_shares - tax_paid)) < 1e-6 * max(1, cash0)
            assert all(tr.cash >= -1e-6 and tr.shares >= 0 for tr in traders.values())

            if res["traded"]:
                last_price = p
            history.append({"period": t, "price": last_price, "volume": res["volume"],
                            "dividend": d, "traded": res["traded"]})
            log("auction_cleared", period=t, price=last_price, traded=res["traded"],
                volume=res["volume"], demand=res["demand"], supply=res["supply"],
                imbalance=res["imbalance"], band=res["band"], fundamental_value=fv, dividend=d,
                tax_collected=tax_paid, n_buy=len(buys), n_sell=len(sells))

            obs = []
            for a, (sh, c, op) in before.items():
                if sh == 0:
                    continue
                sold = any(x["action"] == "sell" for x in trades[a])
                obs.append({"period": t, "agent": a, "agent_persona": agents[a].persona,
                            "stock": "A", "shares": sh, "avg_cost": round(c, 6),
                            "price": last_price, "ret": last_price / c - 1,
                            "gain": None if last_price == c else int(last_price > c),
                            "sold": sold, "sell": int(sold),
                            "qty_sold": sum(x["quantity"] for x in trades[a] if x["action"] == "sell"),
                            "holding_period": t - op, "drawdown": None, "sale_day": int(sold)})
            values = {a: tr.value(last_price) for a, tr in traders.items()}
            log("period_end", period=t, price=last_price, fundamental_value=fv,
                holdings={a: {"cash": tr.cash, "shares": tr.shares, "avg_cost": tr.avg_cost,
                              "value": values[a]} for a, tr in traders.items()},
                orders=orders, observations=obs)

            def end(a):
                agents[a].end_period(t, {"period": t, "value": values[a],
                                         "prev_value": prev_value[a], "trades": trades[a],
                                         "dividend_per_share": d,
                                         "dividend_total": d * traders[a].shares}, T,
                                     dummy_lessons=M.dummy_policy(f"{rep}/{a}", t, traders[a], fv, p,
                                                                  two_sided)[1]
                                     if dummy else None)
            list(pool.map(end, ids))
            prev_value = values
            flat = [o for os in orders.values() for o in os]
            sample = next((o for o in flat if o["action"] != "hold"), flat[0])
            nb = sum(o["action"] == "buy" and o["quantity"] > 0 for o in flat)
            ns = sum(o["action"] == "sell" and o["quantity"] > 0 for o in flat)
            board.update(run_id, period=f"{t}/{T}", price=f"{last_price:.1f}", FV=f"{fv:.0f}",
                         volume=res["volume"], orders=f"{nb} buy / {ns} sell",
                         reason=f"{sample['action']}: {sample['reason'][:70]}")
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
        # The market cannot clear without that agent's order and holds are not allowed,
        # so the whole market run stops (logged by the agent); other runs continue.
        status = "incomplete_stop"
    except (Aborted, Halted):
        status = "aborted"
    finally:
        pool.shutdown(wait=True)
    log("run_end", status=status, final_value=None,
        incomplete=sum(ag.n_incomplete for ag in agents.values()),
        fallbacks=sum(ag.n_fallback for ag in agents.values()),
        json_invalid=sum(ag.n_invalid for ag in agents.values()))
    log.close()
    board.finish(run_id)
    return {"run_id": run_id, "status": status, "fatal": fatal}


def main():
    ap = base_args(__doc__)
    args = ap.parse_args()
    cfg = load_config("study2")
    if args.backend:
        cfg["llm"]["backend"] = args.backend
    backend = cfg["llm"]["backend"]
    # Design v1 (one order per period) keeps the plain study2* names; v2 -> study2_v2*.
    design = cfg.get("design", "v1")
    variant = "" if design == "v1" else design
    name = output_name("study2", backend, args.dry_run, args.stage, variant)
    proj_file = ROOT / "results" / output_name("study2", backend, False, "pilot", variant) \
        / "pilot_projection.json"
    log_name = replay_setup(args, backend, name)
    if backend == "openai" and not (args.dry_run or args.replay):
        require_api_key()      # from .env (git-ignored) or the environment; fails fast if missing
    if args.stage == "full" and not confirm_full_run(cfg, args, proj_file):
        return
    for old in [*(ROOT / "logs" / log_name).glob("run_*.jsonl"),
                *(ROOT / "logs" / log_name).glob("FATAL_ERROR.json")]:
        old.unlink()

    jobs = [(c["name"], cell_params(cfg, c), r)
            for c in cfg["stages"][args.stage]["cells"] for r in range(c.get("runs", 1))]
    if args.dry_run:
        jobs = jobs[:1]
    stop = threading.Event()
    board = Board(f"Study 2 [{name}, {args.stage}]",
                  ["period", "price", "FV", "volume", "orders", "reason"], None,
                  enabled=not args.no_live and not args.dry_run)
    llm = make_llm(cfg, name, args.dry_run, args.show_prompts, echo=board.echo,
                   offline=args.replay, retry_incomplete=args.retry_incomplete)
    board.cost, board.total = llm.cost, len(jobs)
    aw = cfg["run"]["agent_workers"]
    with board, ThreadPoolExecutor(max_workers=args.workers or cfg["run"]["workers"]) as ex:
        results = list(ex.map(lambda j: run_market(cfg, llm, log_name, j[0], j[1], j[2], stop,
                                                   board, aw), jobs))
    print(f"\nruns: { {s: sum(r['status'] == s for r in results) for s in {r['status'] for r in results}} }")
    print(f"cost: {json.dumps(llm.cost.summary())}")
    if args.dry_run:
        p = llm.projection
        n = full_calls(cfg)
        per = llm.cost.price({"input_tokens": p["input_tokens"] / max(1, p["calls"]),
                              "output_tokens": cfg["llm"]["max_output_tokens"]})
        print(f"\nDry run: {p['calls']} prompts built. Upper-bound projection for the full stage "
              f"({n:,} calls at max_output_tokens): ≈ ${n * per:.2f}; cap ${cfg['budget']['cap_usd']:.2f}.")
        return
    if args.replay and (code := finish_replay(results, name, log_name)):
        sys.exit(code)
    dfs = dataset.build(name)
    out = ROOT / "results" / name
    for k, df in dfs.items():
        dataset._save(df, out, k)
    mk = dfs.get("markets")
    if mk is not None and not mk.empty:
        cols = ["run_id", "status", "traded_periods", "no_trade_share", "RAD", "RD",
                "peak_overpricing", "turnover", "volatility", "DE", "incomplete"]
        # reindex: markets stopped before their first auction have no outcome columns
        print("\n" + mk.reindex(columns=cols).round(3).to_string(index=False))
        by_cell = mk.groupby("cell")[["llm_calls", "incomplete", "incomplete_stops",
                                      "invalid_replies", "fallbacks"]].sum()
        print("\nPer cell:\n" + by_cell.to_string())
    if backend == "openai" and not args.replay and llm.cost.session["api_calls"]:
        from llm_investors.sim.cache_stats import reminder
        if (msg := reminder("study2")):
            print("\n" + msg)
    code = report_fatal(results, ROOT / "logs" / name)
    if code:
        sys.exit(code)        # no cost projection from an interrupted pilot
    if args.stage == "pilot" and backend == "openai" and not args.replay:
        st = call_stats(ROOT / "logs" / name, llm.cost.price)
        if st["calls"]:
            check_service_tier(st, cfg["llm"].get("service_tier"))
            calls = round(full_calls(cfg) * (1 + st["json_retry_rate"]))   # incl. JSON retries
            cost = calls * st["mean_cost_usd"]
            proj = {**st, "projected_full_calls": calls, "projected_full_cost_usd": cost,
                    "projected_remaining_cost_usd": cost, "cap_usd": cfg["budget"]["cap_usd"]}
            proj_file.parent.mkdir(parents=True, exist_ok=True)
            proj_file.write_text(json.dumps(proj, indent=2))
            print(f"\nMeasured ${st['mean_cost_usd']:.6f}/call; projected full run {calls:,} calls "
                  f"≈ ${cost:.2f} vs cap ${cfg['budget']['cap_usd']:.2f}. Confirm before --stage full.")


def full_calls(cfg) -> int:
    n = 0
    for c in cfg["stages"]["full"]["cells"]:
        P = cell_params(cfg, c)
        refl = len([r for r in cfg["agents"]["reflect_after"] if r < P["periods"]]) \
            if P["memory"] == "reflect" else 0
        n += c.get("runs", 1) * P["n_agents"] * (P["periods"] + refl)
    return n


if __name__ == "__main__":
    main()
