"""Forecast evaluation metrics and the Diebold-Mariano test.

Price-space metrics (per horizon h and overall):
    MAE  = mean |C_hat - C|                      (EUR)
    RMSE = sqrt(mean (C_hat - C)^2)              (EUR)
    MAPE = 100 * mean |C_hat - C| / C            (%)
Scale-free metrics (log space, e = log C_hat - log C):
    RMSE_log = 100 * sqrt(mean e^2)              (~ % error)
    U        = RMSE_log(model) / RMSE_log(persistence)   (Theil's U2; <1 beats naive)
    DA       = % of windows where sign(C_hat - C_5) = sign(C - C_5)
               (windows with no actual change are excluded; undefined for a
                forecast that is identically zero, i.e. persistence)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def price_metrics(p_hat: np.ndarray, p_true: np.ndarray, last: np.ndarray) -> dict:
    """p_hat, p_true: (N, H) prices; last: (N,) last observed close C_5."""
    err = p_hat - p_true
    e_log = np.log(p_hat) - np.log(p_true)
    move_hat = np.sign(p_hat - last[:, None])
    move = np.sign(p_true - last[:, None])
    out = {}
    for h in range(p_hat.shape[1] + 1):             # h = 0 means "all horizons"
        sl = slice(None) if h == 0 else slice(h - 1, h)
        key = "all" if h == 0 else f"h{h}"
        valid = move[:, sl] != 0
        da = (np.mean((move_hat[:, sl] == move[:, sl])[valid]) * 100
              if np.any(move_hat[:, sl] != 0) else np.nan)
        out[key] = dict(
            MAE=float(np.mean(np.abs(err[:, sl]))),
            RMSE=float(np.sqrt(np.mean(err[:, sl] ** 2))),
            MAPE=float(100 * np.mean(np.abs(err[:, sl]) / p_true[:, sl])),
            RMSE_log=float(100 * np.sqrt(np.mean(e_log[:, sl] ** 2))),
            DA=float(da),
        )
    return out


def metrics_frame(results: dict[str, dict]) -> pd.DataFrame:
    """{model: price_metrics(...)} -> tidy DataFrame (model, horizon, metric...)."""
    rows = []
    for model, m in results.items():
        for hz, vals in m.items():
            rows.append(dict(model=model, horizon=hz, **vals))
    df = pd.DataFrame(rows)
    ref = df[df.model == "Persistence"].set_index("horizon")["RMSE_log"]
    df["U"] = df["RMSE_log"] / df["horizon"].map(ref)
    return df


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, origin: np.ndarray,
                    lag: int = C.WINDOW - 1) -> tuple[float, float, float]:
    """Diebold-Mariano (1995) test of equal predictive accuracy, H0: E[L_a - L_b] = 0.

    * loss_a, loss_b: per-window losses (N,) of two forecasters on the SAME windows.
    * Panel handling: the loss differential is first averaged across stocks for
      each forecast origin date (removes cross-sectional dependence caused by
      the common market factor), giving one time series d_t.
    * The long-run variance of d_t is estimated with a Newey-West (Bartlett)
      kernel with `lag` = 9 lags, which covers the overlap of consecutive
      10-session windows (conservative).
    * Harvey-Leybourne-Newbold small-sample correction, Student-t(T-1) p-value.

    Returns (mean differential, HLN-corrected DM statistic, two-sided p-value).
    Negative statistic => forecaster A is more accurate.
    """
    d = pd.Series(loss_a - loss_b).groupby(pd.Index(origin)).mean().sort_index().values
    T = len(d)
    dbar = d.mean()
    dc = d - dbar
    lrv = dc @ dc / T
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    dm = dbar / np.sqrt(lrv / T)
    h = lag + 1
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(dbar), float(stat), float(p)


def log_sq_loss(y_hat: np.ndarray, y_true: np.ndarray, horizon: int | None = None) -> np.ndarray:
    """Per-window squared log-error, averaged over horizons (or at one horizon)."""
    e2 = (np.asarray(y_hat) - np.asarray(y_true)) ** 2
    return e2.mean(1) if horizon is None else e2[:, horizon - 1]
