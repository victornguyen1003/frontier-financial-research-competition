"""Run the persona x memory-mode x price-path grid and compute disposition metrics.

    python sim.py                    # real run with the configured model
    python sim.py --backend mock     # offline dry run (no API calls)
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent import Agent
from config import Config
from llm import LLM
from market import make_paths, stock_names
import metrics


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, rec: dict):
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            self.f.flush()

    def close(self):
        self.f.close()


def build_state(t, T, cash, pos, prices, labels):
    positions = {}
    for i, s in enumerate(labels):
        sh, cost, px = pos[s]["shares"], pos[s]["cost"], float(prices[t, i])
        positions[s] = {
            "shares": sh,
            "purchase_price": round(cost, 2) if sh > 0 else None,
            "current_price": px,
            "unrealized_pnl_pct": round((px / cost - 1) * 100, 2) if sh > 0 else None,
        }
    return {"period": t, "total_periods": T, "cash": round(cash, 2), "positions": positions}


def portfolio_value(cash, pos, prices_t, labels):
    return cash + sum(pos[s]["shares"] * float(prices_t[i]) for i, s in enumerate(labels))


def run_one(cfg: Config, llm: LLM, persona_idx: int, persona: str, mode: str,
            path_idx: int, prices) -> dict:
    labels = stock_names(cfg.n_stocks)
    run_id = f"persona{persona_idx}_{mode}_path{path_idx}"
    log = JsonlLogger(Path(cfg.out_dir) / f"{run_id}.jsonl")
    log({"type": "run_meta", "run_id": run_id, "persona": persona, "memory_mode": mode,
         "path_idx": path_idx, "path_seed": cfg.seed + path_idx, "config": cfg.to_dict(),
         "prices": prices.tolist()})

    agent = Agent(persona, mode, llm, labels, cfg.reflect_every, cfg.max_json_retries, log)
    # Initial endowment: shares of every stock bought at the period-0 price.
    cash = cfg.initial_cash
    pos = {s: {"shares": cfg.initial_shares, "cost": float(prices[0, i])}
           for i, s in enumerate(labels)}
    observations, clipped, fallbacks, prev_value = [], 0, 0, None

    for t in range(1, cfg.T + 1):
        px = {s: float(prices[t, i]) for i, s in enumerate(labels)}
        value = portfolio_value(cash, pos, prices[t], labels)
        state = build_state(t, cfg.T, cash, pos, prices, labels)
        before = {s: dict(v) for s, v in pos.items() if v["shares"] > 0}
        decision = agent.decide(state)
        fallbacks += bool(decision.get("fallback"))

        # Execute sells first (their proceeds can fund buys), then buys; clip infeasible orders.
        executed, realized, sold = [], [], set()
        order = sorted(decision["trades"], key=lambda tr: tr["action"] != "sell")
        for tr in order:
            s, a, q = tr["stock"], tr["action"], tr["shares"]
            if a == "sell":
                q_exec = min(q, pos[s]["shares"])
                if q_exec > 0:
                    cash += q_exec * px[s]
                    pct = (px[s] / pos[s]["cost"] - 1) * 100
                    realized.append(f"sold {q_exec} {s} at {pct:+.1f}% vs purchase price")
                    pos[s]["shares"] -= q_exec
                    sold.add(s)
            elif a == "buy":
                q_exec = min(q, int(cash // px[s]))
                if q_exec > 0:
                    sh, cost = pos[s]["shares"], pos[s]["cost"]
                    pos[s]["cost"] = (sh * cost + q_exec * px[s]) / (sh + q_exec) if sh else px[s]
                    pos[s]["shares"] += q_exec
                    cash -= q_exec * px[s]
            else:
                q_exec = 0
            clipped += q_exec != q
            executed.append({"stock": s, "action": a, "requested": q, "shares": q_exec, "price": px[s]})

        # Odean observations: every position held before this period's trades.
        obs = [{"period": t, "stock": s, "price": px[s], "cost": round(st["cost"], 6),
                "sold": s in sold} for s, st in before.items()]
        observations += obs
        log({"type": "trades", "period": t, "decision": decision, "executed": executed,
             "cash_after": cash, "value_before_trades": value})
        log({"type": "positions", "period": t, "observations": obs})

        agent.record(t, cfg.T, executed, value, prev_value, realized)
        prev_value = value

    final_value = portfolio_value(cash, pos, prices[cfg.T], labels)
    m = metrics.run_metrics(observations)
    result = {"run_id": run_id, "persona": persona, "memory_mode": mode, "path_idx": path_idx,
              **m, "json_failures": agent.json_failures, "fallback_periods": fallbacks,
              "clipped_orders": clipped, "final_value": round(final_value, 2),
              "lessons": agent.lessons}
    log({"type": "result", **result})
    log.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "mock"])
    ap.add_argument("--model")
    ap.add_argument("--paths", type=int)
    ap.add_argument("--modes", nargs="+", choices=["none", "log", "reflect"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    cfg = Config()
    for k, v in {"backend": args.backend, "model": args.model, "n_paths": args.paths,
                 "memory_modes": args.modes, "seed": args.seed, "workers": args.workers}.items():
        if v is not None:
            setattr(cfg, k, v)
    if cfg.backend == "mock":
        cfg.model = "mock"

    llm = LLM(cfg)
    paths = make_paths(cfg)
    grid = list(itertools.product(enumerate(cfg.personas), cfg.memory_modes, range(cfg.n_paths)))
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        results = list(ex.map(lambda g: run_one(cfg, llm, g[0][0], g[0][1], g[1], g[2], paths[g[2]]),
                              grid))

    f = lambda x: "  n/a" if x is None else f"{x:+.2f}" if x < 0 else f"{x:.3f}"
    print(f"\n{'run':<28}{'RG':>4}{'RL':>4}{'PG':>4}{'PL':>4}{'PGR':>7}{'PLR':>7}{'DE':>7}"
          f"{'jsonFail':>9}{'clip':>5}")
    for r in results:
        print(f"{r['run_id']:<28}{r['RG']:>4}{r['RL']:>4}{r['PG']:>4}{r['PL']:>4}"
              f"{f(r['PGR']):>7}{f(r['PLR']):>7}{f(r['DE']):>7}{r['json_failures']:>9}"
              f"{r['clipped_orders']:>5}")
    summary = {"config": cfg.to_dict(), "llm_stats": llm.stats, "runs": results, "by_mode": {}}
    print("\nPooled by memory mode (counts summed over paths):")
    for (pi, persona), mode in itertools.product(enumerate(cfg.personas), cfg.memory_modes):
        rs = [r for r in results if r["persona"] == persona and r["memory_mode"] == mode]
        p = metrics.pooled(rs)
        des = [r["DE"] for r in rs if r["DE"] is not None]
        p["mean_run_DE"] = sum(des) / len(des) if des else None
        p["json_failures"] = sum(r["json_failures"] for r in rs)
        summary["by_mode"][f"persona{pi}_{mode}"] = p
        print(f"  persona{pi} {mode:<8} PGR={f(p['PGR'])} PLR={f(p['PLR'])} DE={f(p['DE'])} "
              f"mean-run DE={f(p['mean_run_DE'])} json_failures={p['json_failures']}")
    print(f"\nLLM: {llm.stats}")
    out = Path(cfg.out_dir) / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Logs: {cfg.out_dir}/*.jsonl   Summary: {out}")


if __name__ == "__main__":
    main()
