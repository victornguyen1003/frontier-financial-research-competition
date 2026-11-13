"""Memory architectures: none | log | reflect.

none     current-period information only
log      + factual history of all past periods (value, % change, trades, realized P&L vs cost).
         This is the agent's own trading record, not market memory: it never repeats past
         prices. (Study 2 shows the public price history to every architecture.)
reflect  + 3 self-written lessons from the latest reflection (after periods 5 and 10);
         periods before the first reflection are identical to `none`.
The raw history is kept for every architecture (reflect needs it as reflection input).
"""
from __future__ import annotations

ARCHITECTURES = ("none", "log", "reflect")


def format_trade(t: dict, show_pnl: bool = True) -> str:
    verb = "sold" if t["action"] == "sell" else "bought"
    s = f"{verb} {t['quantity']} of Stock {t['stock']} at {t['price']:,.2f}"
    if show_pnl and t["action"] == "sell" and t.get("avg_cost"):
        s += f" ({(t['price'] / t['avg_cost'] - 1) * 100:+.1f}% vs your average cost)"
    return s


def history_line(rec: dict, show_pnl: bool = True) -> str:
    """rec: {period, value, prev_value, trades, dividend_per_share?, dividend_total?}"""
    line = f"Period {rec['period']}: account value {rec['value']:,.2f}"
    if rec.get("prev_value"):
        line += f" ({(rec['value'] / rec['prev_value'] - 1) * 100:+.1f}% since the previous period)"
    trades = [t for t in rec.get("trades", []) if t["quantity"] > 0]
    line += ". You " + "; ".join(format_trade(t, show_pnl) for t in trades) if trades else \
        ". You made no trades"
    if rec.get("dividend_per_share") is not None:
        line += (f". Dividend: {rec['dividend_per_share']:g} per share "
                 f"({rec.get('dividend_total', 0):,.2f} received)")
    return line + "."


class Memory:
    def __init__(self, architecture: str, show_pnl: bool = True):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        self.architecture = architecture
        self.show_pnl = show_pnl
        self.history: list[str] = []
        self.lessons: list[str] = []

    def add(self, rec: dict):
        self.history.append(history_line(rec, self.show_pnl))

    def block(self) -> str | None:
        if self.architecture == "log" and self.history:
            return "Your record of previous periods:\n" + "\n".join(self.history)
        if self.architecture == "reflect" and self.lessons:
            return ("Lessons you wrote from your own experience so far:\n" +
                    "\n".join(f"{i}. {x}" for i, x in enumerate(self.lessons, 1)))
        return None
