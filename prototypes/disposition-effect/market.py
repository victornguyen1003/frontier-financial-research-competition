"""Geometric Brownian motion price paths for anonymous stocks."""
from __future__ import annotations

import numpy as np


def stock_names(n: int) -> list[str]:
    """Anonymous labels: A, B, C, ... (shown to the agent as 'Stock A', ...)."""
    return [chr(ord("A") + i) for i in range(n)]


def gbm_path(n_stocks: int, T: int, mu, sigma, s0: float, seed: int) -> np.ndarray:
    """Return prices with shape (T + 1, n_stocks); row 0 is the purchase price.

    Uses the exact GBM step with dt = 1 period:
        S_{t+1} = S_t * exp((mu - sigma^2 / 2) + sigma * Z),  Z ~ N(0, 1)
    The same seed always yields the same path.
    """
    rng = np.random.default_rng(seed)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n_stocks,))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n_stocks,))
    z = rng.standard_normal((T, n_stocks))
    log_ret = (mu - 0.5 * sigma**2) + sigma * z
    log_path = np.vstack([np.zeros(n_stocks), np.cumsum(log_ret, axis=0)])
    return np.round(s0 * np.exp(log_path), 2)


def make_paths(cfg) -> list[np.ndarray]:
    """One path per index k, seeded with cfg.seed + k."""
    return [gbm_path(cfg.n_stocks, cfg.T, cfg.mu, cfg.sigma, cfg.s0, cfg.seed + k)
            for k in range(cfg.n_paths)]


if __name__ == "__main__":
    from config import Config
    for k, p in enumerate(make_paths(Config())):
        print(f"path {k}: final prices {dict(zip(stock_names(p.shape[1]), p[-1]))}")
