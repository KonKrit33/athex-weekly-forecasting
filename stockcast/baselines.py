"""Classical (non-neural) reference predictors.

All functions return forecasts in the scaled log-ratio space used everywhere:
    y_hat_h = log(C_hat_{5+h} / C_5) / s,   h = 1..5
so that they are evaluated with exactly the same code as the neural models.

* Persistence (naive / random-walk forecast): C_hat_{5+h} = C_5.
* RW + drift (TSA notes, x_t = delta + x_{t-1} + w_t on log-prices): drift
  delta estimated once on the training stocks.
* Linear trend (least squares, Lectures 4-5): straight line fitted to the 5
  log-closes of the input week and extrapolated.
* SES (simple exponential smoothing, Lectures 4-5): m_t = a x_t + (1-a) m_{t-1}
  on the log-closes of the week; flat forecast m_5; a chosen on the train set.
* AR(4)-OLS: linear autoregression of order 4 on daily log-returns fitted by
  ordinary least squares on the training stocks (as ar.ols in the solved
  problems), iterated recursively for h = 1..5 - the linear counterpart of the
  autoregressive Vanilla RNN.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .data import Scaler, WindowSet, ar_context, direct_targets, log_returns


def persistence(w: WindowSet, sc: Scaler) -> np.ndarray:
    return np.zeros((len(w), C.H_OUT), dtype="float64")


def rw_drift(w: WindowSet, sc: Scaler) -> np.ndarray:
    h = np.arange(1, C.H_OUT + 1)
    return np.tile(h * sc.mu_ret / sc.s_ret, (len(w), 1))


def linear_trend(w: WindowSet, sc: Scaler) -> np.ndarray:
    t = np.arange(1, C.L_IN + 1, dtype="float64")
    x = np.log(w.close[:, :C.L_IN])
    tc = t - t.mean()
    slope = (x - x.mean(1, keepdims=True)) @ tc / (tc @ tc)
    intercept = x.mean(1) - slope * t.mean()
    t_future = np.arange(C.L_IN + 1, C.L_IN + C.H_OUT + 1)
    level = intercept[:, None] + slope[:, None] * t_future[None, :]
    return (level - x[:, -1:]) / sc.s_ret


def _ses_level(x: np.ndarray, alpha: float) -> np.ndarray:
    m = x[:, 0].copy()
    for t in range(1, x.shape[1]):
        m = alpha * x[:, t] + (1 - alpha) * m
    return m


def ses_fit(w_train: WindowSet, sc: Scaler) -> float:
    """Choose the smoothing constant on the TRAIN windows (grid search, MSE)."""
    x = np.log(w_train.close[:, :C.L_IN])
    y = direct_targets(w_train, sc)
    best = None
    for a in np.linspace(0.05, 1.0, 96):
        yhat = (_ses_level(x, a) - x[:, -1])[:, None] / sc.s_ret
        mse = float(np.mean((y - yhat) ** 2))
        if best is None or mse < best[1]:
            best = (float(a), mse)
    return best[0]


def ses(w: WindowSet, sc: Scaler, alpha: float) -> np.ndarray:
    x = np.log(w.close[:, :C.L_IN])
    yhat = (_ses_level(x, alpha) - x[:, -1]) / sc.s_ret
    return np.tile(yhat[:, None], (1, C.H_OUT))


class AROLS:
    """AR(p) on scaled daily log-returns, fitted by OLS, iterated forecasts."""

    def __init__(self, p: int = C.L_IN - 1):
        self.p = p
        self.coef = None

    def fit(self, w_train: WindowSet, sc: Scaler) -> "AROLS":
        r = log_returns(w_train, sc)                   # (N, 9): r_2..r_10
        X, y = [], []
        for t in range(self.p, r.shape[1]):            # every one-step pair in the window
            X.append(r[:, t - self.p:t])
            y.append(r[:, t])
        X = np.concatenate(X)
        y = np.concatenate(y)
        X = np.column_stack([np.ones(len(X)), X])
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def forecast(self, w: WindowSet, sc: Scaler) -> np.ndarray:
        hist = ar_context(w, sc)[:, :, 0].astype("float64")   # r_2..r_5
        preds = []
        for _ in range(C.H_OUT):
            lags = hist[:, -self.p:]
            nxt = self.coef[0] + lags @ self.coef[1:]
            preds.append(nxt)
            hist = np.column_stack([hist, nxt])
        return np.cumsum(np.column_stack(preds), axis=1)
