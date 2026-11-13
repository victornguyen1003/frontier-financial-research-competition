import numpy as np
from llm_investors.envs.market import Trader, clear_call_auction, validate


def b(i, q, p):
    return {"id": i, "quantity": q, "limit_price": p}


def test_volume_maximizing_price():
    # demand: 3 @ >=105, +2 @ >=100 ; supply: 2 @ <=98, +3 @ <=102
    buys = [b("b1", 3, 105), b("b2", 2, 100)]
    sells = [b("s1", 2, 98), b("s2", 3, 102)]
    r = clear_call_auction(buys, sells, ref_price=100)
    # p=100: D=5, S=2 -> 2 ; p=102: D=3, S=5 -> 3 ; p=105: D=3, S=5 -> 3 ; p=98: D=5,S=2 -> 2
    # max volume 3 at 102 or 105 -> closest to ref 100 -> 102
    assert r["price"] == 102 and r["volume"] == 3
    assert r["fills"]["b1"] == 3 and "b2" not in r["fills"]
    # supply side is long (5 vs 3): pro rata 2/5*3=1.2 -> 1, 3/5*3=1.8 -> 2
    assert r["fills"]["s1"] == -1 and r["fills"]["s2"] == -2
    assert sum(r["fills"].values()) == 0


def test_no_cross():
    r = clear_call_auction([b("b", 1, 90)], [b("s", 1, 95)], ref_price=92)
    assert not r["traded"] and r["volume"] == 0 and r["price"] == 92 and r["fills"] == {}


def test_tie_break_prefers_reference():
    # volume 1 anywhere in [90, 110] -> clears at the reference price itself
    r = clear_call_auction([b("b", 1, 110)], [b("s", 1, 90)], ref_price=100)
    assert r["price"] == 100 and r["volume"] == 1
    # reference outside the crossing range -> nearest crossing price
    r = clear_call_auction([b("b", 1, 110)], [b("s", 1, 104)], ref_price=100)
    assert r["price"] == 104


def test_price_limit_band():
    # buyer would pay 200, seller wants >= 150: they cross only above the +10% band -> no trade
    r = clear_call_auction([b("b", 2, 200)], [b("s", 2, 150)], ref_price=100, price_limit=0.10)
    assert not r["traded"] and r["band"] == [90.0, 110.0]
    # buyer 200, seller 105: crossing range [105, 200] meets the band at [105, 110] -> 105
    r = clear_call_auction([b("b", 2, 200)], [b("s", 2, 105)], ref_price=100, price_limit=0.10)
    assert r["price"] == 105 and r["volume"] == 2
    # buyer 200, seller 50: clips to the band; reference 100 is inside -> 100
    r = clear_call_auction([b("b", 2, 200)], [b("s", 2, 50)], ref_price=100, price_limit=0.10)
    assert r["price"] == 100


def test_imbalance_tiebreak_moves_price_under_excess_supply():
    # 2 shares wanted at <= 370; 30 offered between 250 and 330; previous price 360.
    buys = [b("b", 2, 370)]
    sells = [b(f"s{i}", 2, 250 + 5 * i) for i in range(15)]
    sticky = clear_call_auction(buys, sells, ref_price=360, min_imbalance=False)
    moving = clear_call_auction(buys, sells, ref_price=360, min_imbalance=True)
    assert sticky["price"] == 360            # proposal's literal rule: price cannot fall
    assert moving["price"] == 250 and moving["volume"] == 2 and moving["imbalance"] == 0


def test_pro_rata_largest_remainder():
    buys = [b("b1", 1, 100), b("b2", 1, 100), b("b3", 1, 100)]
    r = clear_call_auction(buys, [b("s", 2, 100)], ref_price=100)
    assert r["volume"] == 2 and sum(q for k, q in r["fills"].items() if k != "s") == 2


def test_validate_clips_to_cash_and_shares():
    tr = Trader(cash=250, shares=1, avg_cost=100)
    o, rej = validate({"action": "buy", "quantity": 5, "limit_price": 100}, tr)
    assert o["quantity"] == 2 and rej["executed"] == 2
    o, rej = validate({"action": "sell", "quantity": 3, "limit_price": 100}, tr)
    assert o["quantity"] == 1 and rej["requested"] == 3


def test_bubble_measures_ignore_no_trade_periods():
    from llm_investors.measure.market_metrics import bubble_measures, trading_measures
    fv = [240, 216, 192, 168]
    # design v1 pilot: one trade at 240 in period 1, then a frozen carried-forward price
    frozen = bubble_measures([240] * 4, fv, [True, False, False, False])
    assert frozen["traded_periods"] == 1 and frozen["no_trade_share"] == 0.75
    assert frozen["RD"] == 0.0 and frozen["peak_period"] == 1        # not a bubble
    assert bubble_measures([240] * 4, fv)["RD"] > 0.1                 # what v1 metrics reported
    none = bubble_measures([240] * 4, fv, [False] * 4)
    assert none["traded_periods"] == 0 and np.isnan(none["RAD"]) and none["peak_period"] is None
    part = bubble_measures([240, 240, 150, 150], fv, [True, False, True, False])
    assert part["peak_period"] == 1 and abs(part["crash"] - (1 - 150 / 240)) < 1e-12
    # returns skip the carried-forward prices: 240 -> 150 -> 120 is two returns, not three
    t = trading_measures([240, 240, 150, 120], [3, 0, 2, 1], 38, [True, False, True, True])
    assert t["volume_total"] == 6 and t["volatility"] is None        # < 3 returns
