"""Reasoning tags (plan §4.2): `tag-reasoning <dataset>`.

An LLM classifier with a fixed prompt returns one boolean per tag for each `reasoning` text.
Calls go through the same client, disk cache (cache/study1/, so the overall budget cap covers
them) and stage limit (budget.stage_limits.tagging) as the agents. Texts are de-duplicated
and sent `tagging.batch_size` at a time in a fixed order, so an identical re-run replays from
the cache at $0.

    tag-reasoning study1_v3_gpt-6-luna_pilot --dry-run      # projection, no API
    tag-reasoning study1_v3_gpt-6-luna_pilot                # paid, asks first
    tag-reasoning study1_v3_gpt-6-luna_pilot_dummy --backend dummy

Writes results/<dataset>/reasoning_tags.{csv,parquet} and, once, a hand-coding sample
reasoning_validation.csv (100 texts). Fill its human_* columns with 0/1; the next run reports
agreement and Cohen's kappa per tag in reasoning_agreement.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from llm_investors.agents.parsing import InvalidReply, extract_json
from llm_investors.llm.client import FatalAPIError, IncompleteLimit
from llm_investors.llm.cost import BudgetExceeded, price_usage
from llm_investors.paths import ROOT
from llm_investors.sim.common import load_config, make_llm

TAGS = ["persona", "reference_point", "past_trades", "forecast", "randomness"]

CLASSIFIER_SYSTEM = """You label short texts in which an investor explains a monthly trading \
decision. For each text, answer five yes/no questions. A label is true only if the text \
clearly refers to it; do not infer.

persona: the text refers to the investor's own risk attitude, feelings about gains or losses, \
experience, profession, age or trading habits.
reference_point: the text refers to a gain or loss relative to the purchase price or average \
cost of a position (for example "up 8% since I bought it", "below my cost").
past_trades: the text refers to the investor's own earlier trades, earlier decisions or \
lessons learned from them.
forecast: the text states or implies an expectation about future prices (for example "likely \
to recover", "momentum should continue", "may fall further").
randomness: the text says that prices are random or unpredictable, or that past price changes \
do not predict future changes.

Reply with ONLY a compact JSON object, no other text:
{"labels":[{"id":<id>,"persona":true|false,"reference_point":true|false,\
"past_trades":true|false,"forecast":true|false,"randomness":true|false}, ...]}
Give exactly one entry for every id."""


def user_prompt(batch: list[tuple[int, str]]) -> str:
    items = "\n".join(json.dumps({"id": i, "text": t}, ensure_ascii=False) for i, t in batch)
    return f"Texts:\n{items}\n\nLabel every text. Reply with the JSON object only."


def parse_labels(text: str, ids: list[int]) -> dict[int, dict]:
    obj = extract_json(text)
    labs = obj.get("labels")
    if not isinstance(labs, list):
        raise InvalidReply('"labels" must be a list')
    out = {}
    for e in labs:
        if not isinstance(e, dict) or e.get("id") not in ids:
            raise InvalidReply(f"unknown or missing id in {e!r}")
        row = {}
        for t in TAGS:
            v = e.get(t)
            if not isinstance(v, bool):
                raise InvalidReply(f'"{t}" must be true or false for id {e.get("id")}')
            row[t] = v
        out[e["id"]] = row
    missing = [i for i in ids if i not in out]
    if missing:
        raise InvalidReply(f"missing ids {missing}; give exactly one entry for every id")
    return out


# --------------------------------------------------------------------------- dummy labels
RULES = {
    "persona": r"\b(as an?|i am|i'm|my (risk|experience|age)|experienced|professional|"
               r"years old|i (dislike|hate|enjoy)|painful|satisf)",
    "reference_point": r"(since i bought|vs (my|the) (average )?cost|above (my )?cost|"
                       r"below (my )?cost|up \d|down \d|\bgain\b|\bloss\b)",
    "past_trades": r"(last month|earlier|previous|lesson|i sold|i bought|my record)",
    "forecast": r"(expect|likely|should (rise|fall|recover)|will (rise|fall)|rebound|recover)",
    "randomness": r"(random|unpredictab|do not predict|don't predict)",
}


def dummy_labels(batch: list[tuple[int, str]]) -> str:
    return json.dumps({"labels": [{"id": i, **{t: bool(re.search(p, s.lower()))
                                               for t, p in RULES.items()}} for i, s in batch]})


# --------------------------------------------------------------------------- client
def tagging_cfg(cfg: dict, backend: str | None) -> tuple[dict, float | None]:
    t = cfg["tagging"]
    c = {**cfg, "llm": {**cfg["llm"]}}
    model = t["model"]
    prices = (cfg["llm"].get("model_prices") or {}).get(model, cfg["llm"]["price_per_mtok"])
    prices = dict(prices)
    if "service_tier" in prices:
        c["llm"]["service_tier"] = prices.pop("service_tier")
    c["llm"].update(model=model, reasoning_effort=t.get("reasoning_effort"),
                    max_output_tokens=t["max_output_tokens"], price_per_mtok=prices)
    if backend:
        c["llm"]["backend"] = backend
    return c, (cfg["budget"].get("stage_limits") or {}).get("tagging")


def batches(texts: list[str], size: int) -> list[list[tuple[int, str]]]:
    """Fixed order (by content hash) so the same texts always form the same requests."""
    order = sorted(range(len(texts)), key=lambda i: hashlib.sha256(texts[i].encode()).hexdigest())
    idx = [(i, texts[i]) for i in order]
    return [idx[k:k + size] for k in range(0, len(idx), size)]


def estimate(cfg: dict, bs: list) -> dict:
    chars = sum(len(CLASSIFIER_SYSTEM) + len(user_prompt(b)) for b in bs)
    out_tok = sum(40 * len(b) + 300 for b in bs)        # labels + low-effort reasoning
    upper_out = len(bs) * cfg["llm"]["max_output_tokens"]
    p = cfg["llm"]["price_per_mtok"]
    return {"calls": len(bs), "input_tokens": chars // 4,
            "typical_usd": price_usage(p, {"input_tokens": chars // 4, "output_tokens": out_tok}),
            "upper_usd": price_usage(p, {"input_tokens": chars // 4, "output_tokens": upper_out})}


def classify(llm, batch, dummy: bool, retries: int = 1) -> tuple[dict, dict]:
    ids = [i for i, _ in batch]
    messages = [{"role": "user", "content": user_prompt(batch)}]
    err = None
    for _ in range(retries + 1):
        res = llm.complete(CLASSIFIER_SYSTEM, messages,
                           dummy=(lambda: dummy_labels(batch)) if dummy else None)
        try:
            return parse_labels(res["text"], ids), {"error": None, "cost": res["cost_usd"],
                                                    "cached": res["cached"]}
        except InvalidReply as e:
            err = str(e)
            messages = messages + [
                {"role": "assistant", "content": res["text"] or "(empty)"},
                {"role": "user", "content": f"Your reply could not be used: {err}. Reply again "
                                            "with ONLY the JSON object."}]
    return {}, {"error": err, "cost": 0.0, "cached": False}


# --------------------------------------------------------------------------- validation
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a.astype(int), b.astype(int)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return None if pe == 1 else (po - pe) / (1 - pe)


def validation(out_dir, tags: pd.DataFrame, n: int, seed: int = 0) -> dict | None:
    path = out_dir / "reasoning_validation.csv"
    if not path.exists():
        s = tags.dropna(subset=["tag_persona"])
        s = s.sample(min(n, len(s)), random_state=seed)
        v = s[["run_id", "period", "reasoning"]].copy()
        for t in TAGS:
            v[f"human_{t}"] = ""
        v.to_csv(path, index=False)
        print(f"Hand-coding sample written: {path} (fill human_* with 0/1, then re-run)")
        return None
    v = pd.read_csv(path)
    m = v.merge(tags[["run_id", "period"] + [f"tag_{t}" for t in TAGS]], on=["run_id", "period"])
    out = {}
    for t in TAGS:
        h = pd.to_numeric(m[f"human_{t}"], errors="coerce")
        ok = h.notna() & m[f"tag_{t}"].notna()
        if ok.sum() == 0:
            continue
        a, b = h[ok].to_numpy(), m.loc[ok, f"tag_{t}"].astype(float).to_numpy()
        out[t] = {"n": int(ok.sum()), "agreement": float((a == b).mean()),
                  "kappa": cohen_kappa(a, b), "human_rate": float(a.mean()),
                  "model_rate": float(b.mean())}
    if out:
        (out_dir / "reasoning_agreement.json").write_text(json.dumps(out, indent=2))
        print("Agreement with hand coding:", {k: (round(v["agreement"], 3),
                                                 None if v["kappa"] is None else
                                                 round(v["kappa"], 3)) for k, v in out.items()})
    return out or None


# --------------------------------------------------------------------------- main
def main(name: str, backend: str | None = None, dry_run: bool = False, yes: bool = False,
         workers: int = 8):
    cfg0 = load_config("study1")
    cfg, limit = tagging_cfg(cfg0, backend)
    out_dir = ROOT / "results" / name
    p = out_dir / "reasoning.parquet"
    r = pd.read_parquet(p) if p.exists() else pd.read_csv(out_dir / "reasoning.csv")
    r = r.dropna(subset=["reasoning"])
    texts = sorted(r["reasoning"].unique())
    bs = batches(texts, cfg0["tagging"]["batch_size"])
    real = cfg["llm"]["backend"] == "openai" and not dry_run
    est = estimate(cfg, bs)
    print(f"{len(r):,} reasoning texts ({len(texts):,} unique) in {len(bs)} calls to "
          f"{cfg['llm']['model']}: typical ≈ ${est['typical_usd']:.2f}, upper bound "
          f"${est['upper_usd']:.2f}; stage limit ${limit}")
    if dry_run:
        print("=" * 30 + " SYSTEM " + "=" * 30 + "\n" + CLASSIFIER_SYSTEM)
        print("-" * 30 + " USER " + "-" * 30 + "\n" + user_prompt(bs[0][:3]))
        return
    if real:
        from llm_investors.env import require_api_key
        require_api_key()
        if limit is not None and est["typical_usd"] > limit:
            sys.exit("Projection exceeds the tagging stage limit; raise tagging.batch_size or "
                     "tag a subset first.")
        if not yes and input("Start paid tagging? [y/N] ").strip().lower() != "y":
            return
    llm = make_llm(cfg, f"{name}_tags", False, 0, stage_limit=limit if real else None)
    dummy = cfg["llm"]["backend"] == "dummy"
    labels: dict[int, dict] = {}
    errors, stop = [], threading.Event()

    def work(b):
        if stop.is_set():
            return
        try:
            got, info = classify(llm, b, dummy)
        except (BudgetExceeded, FatalAPIError, IncompleteLimit) as e:
            stop.set()
            errors.append(f"{type(e).__name__}: {e}")
            return
        labels.update(got)
        if info["error"]:
            errors.append(info["error"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, bs))
    lab = pd.DataFrame([{"reasoning": texts[i], **{f"tag_{t}": v[t] for t in TAGS}}
                        for i, v in labels.items()])
    if lab.empty:
        lab = pd.DataFrame(columns=["reasoning"] + [f"tag_{t}" for t in TAGS])
    tags = r.merge(lab, on="reasoning", how="left")
    for t in TAGS:
        tags[f"tag_{t}"] = tags[f"tag_{t}"].astype(float)
    tags.to_csv(out_dir / "reasoning_tags.csv", index=False)
    try:
        tags.to_parquet(out_dir / "reasoning_tags.parquet", index=False)
    except Exception:  # noqa: BLE001
        pass
    s = llm.cost.summary()
    print(f"tagged {len(labels):,}/{len(texts):,} unique texts; API calls {s['api_calls']}, "
          f"cache hits {s['cache_hits']}, spent ${s['spent_usd']:.4f}"
          + (f"; {len(errors)} problems, e.g. {errors[0]}" if errors else ""))
    print(tags[[f"tag_{t}" for t in TAGS]].mean().round(3).to_string())
    validation(out_dir, tags, cfg0["tagging"]["validation_sample"])
    print(f"-> {out_dir / 'reasoning_tags.csv'}  (re-run analyze-study1 {name} to use them)")


def cli():
    ap = argparse.ArgumentParser(description="Tag the reasoning texts of a Study 1 v3 dataset")
    ap.add_argument("dataset")
    ap.add_argument("--backend", choices=["openai", "dummy"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.dataset, a.backend, a.dry_run, a.yes, a.workers)


if __name__ == "__main__":
    cli()
