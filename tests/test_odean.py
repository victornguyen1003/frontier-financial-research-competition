from llm_investors.measure.odean import disposition


def ob(period, price, cost, sold):
    return {"period": period, "price": price, "avg_cost": cost, "sold": sold}


# Hand-checked example (build-order step 1 exit check).
#   period 1 (sale day): A sold at a gain -> RG; B held at a loss -> PL; C held at a gain -> PG
#   period 2 (no sale):  nothing counted under Odean's rule
#   period 3 (sale day): B sold at a loss -> RL; C held at a gain -> PG; D at cost -> ignored
# RG=1 RL=1 PG=2 PL=1 -> PGR = 1/3, PLR = 1/2, DE = -1/6
OBS = [
    ob(1, 110, 100, True), ob(1, 90, 100, False), ob(1, 120, 100, False),
    ob(2, 95, 100, False), ob(2, 125, 100, False),
    ob(3, 80, 100, True), ob(3, 130, 100, False), ob(3, 100, 100, False),
]


def test_sale_days_only():
    r = disposition(OBS, sale_days_only=True)
    assert (r["RG"], r["RL"], r["PG"], r["PL"]) == (1, 1, 2, 1)
    assert abs(r["PGR"] - 1 / 3) < 1e-12
    assert abs(r["PLR"] - 1 / 2) < 1e-12
    assert abs(r["DE"] - (-1 / 6)) < 1e-12


def test_all_periods_variant():
    r = disposition(OBS, sale_days_only=False)
    # period 2 adds PL (95 < 100) and PG (125 > 100)
    assert (r["RG"], r["RL"], r["PG"], r["PL"]) == (1, 1, 3, 2)


def test_undefined_when_no_losses():
    r = disposition([ob(1, 110, 100, True), ob(1, 120, 100, False)])
    assert r["PGR"] == 0.5 and r["PLR"] is None and r["DE"] is None


def test_ratio():
    r = disposition([ob(1, 110, 100, True), ob(1, 120, 100, False), ob(1, 90, 100, True),
                     ob(1, 80, 100, False), ob(1, 85, 100, False), ob(1, 70, 100, False)])
    assert r["PGR"] == 0.5 and r["PLR"] == 0.25 and r["Ratio"] == 2.0
