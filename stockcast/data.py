"""Data loading, windowing and (invertible) transformations.

Problem: given one week (5 sessions) of prices of a stock, predict the closing
prices of the following week (5 sessions).

Notation for a window of 10 consecutive sessions of one stock:
    C_1..C_5   closing prices of the input week   (C_5 = last observed close)
    C_6..C_10  closing prices of the target week  (to be predicted)

Transformations (motivated by the EDA, see eda.py and the report):
  * log transform  -> variance stabilisation (multiplicative price dynamics)
  * anchoring on the last observed close, log(C_t / C_5)  -> removes the local
    level / stochastic trend (random walk), so that all stocks, whatever their
    price level, live on the same scale
  * division by a single global constant (std of daily log-returns on the
    training stocks) -> inputs/targets of order 1 for the neural networks

All transformations use only information available at forecast time (the input
week) or constants estimated on the training stocks, therefore no information
from the target week or from the validation/test stocks leaks into training.
The inverse transform is exact: C_hat_{5+h} = C_5 * exp(s * y_hat_h).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_split(split: str, data_dir: Path = C.DATA_DIR) -> dict[int, pd.DataFrame]:
    """Return {stock_id: DataFrame sorted by Date} for one split."""
    frames = {}
    for path in sorted((data_dir / split).glob("*.csv"), key=lambda p: int(p.stem)):
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        if df[["Close", "Open", "High", "Low", "Volume"]].isna().any().any():
            raise ValueError(f"Missing values in {path}")
        frames[int(path.stem)] = df
    if not frames:
        raise FileNotFoundError(f"No csv files found in {data_dir / split}")
    return frames


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
@dataclass
class WindowSet:
    """All (input week, target week) windows of one split."""

    stock: np.ndarray      # (N,)      stock id
    dates: np.ndarray      # (N, 10)   session dates (datetime64[D])
    close: np.ndarray      # (N, 10)   closing prices C_1..C_10
    open: np.ndarray       # (N, 5)    input-week opening prices
    high: np.ndarray       # (N, 5)    input-week highs
    low: np.ndarray        # (N, 5)    input-week lows
    volume: np.ndarray     # (N, 5)    input-week volumes
    n_dropped: int = 0     # windows removed because of trading suspensions
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stock)

    @property
    def last_close(self) -> np.ndarray:           # C_5, (N,)
        return self.close[:, C.L_IN - 1]

    @property
    def target_close(self) -> np.ndarray:         # C_6..C_10, (N, 5)
        return self.close[:, C.L_IN:]

    @property
    def origin(self) -> np.ndarray:               # date of C_5 (forecast origin)
        return self.dates[:, C.L_IN - 1]

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(self.stock[mask], self.dates[mask], self.close[mask],
                         self.open[mask], self.high[mask], self.low[mask],
                         self.volume[mask], self.n_dropped, dict(self.meta))

    def calendar_week_mask(self) -> np.ndarray:
        """True when the input is a full Mon-Fri week and the target is the
        full Mon-Fri week that immediately follows (strict weekly protocol)."""
        days = self.dates.astype("datetime64[D]").view("int64")  # days since 1970-01-01 (Thu)
        wd = (days - 4) % 7    # 0=Mon ... 4=Fri
        wk = (days - 4) // 7   # Monday-based week index
        inp_ok = (wd[:, :5] == np.arange(5)).all(1) & (wk[:, :5] == wk[:, [0]]).all(1)
        out_ok = (wd[:, 5:] == np.arange(5)).all(1) & (wk[:, 5:] == wk[:, [5]]).all(1)
        return inp_ok & out_ok & (wk[:, 5] == wk[:, 0] + 1)


def build_windows(frames: dict[int, pd.DataFrame], stride: int = 1,
                  max_gap_days: int = C.MAX_GAP_DAYS) -> WindowSet:
    """Slide a window of L_IN + H_OUT sessions over every stock."""
    parts = {k: [] for k in ("stock", "dates", "close", "open", "high", "low", "volume")}
    n_dropped = 0
    for sid, df in frames.items():
        n = len(df)
        if n < C.WINDOW:
            continue
        dates = df["Date"].values.astype("datetime64[D]")
        gaps = np.diff(dates).astype(int)                      # calendar days
        idx = np.arange(0, n - C.WINDOW + 1, stride)
        win = idx[:, None] + np.arange(C.WINDOW)[None, :]
        # a window is valid if no gap inside it exceeds max_gap_days
        bad_gap = gaps > max_gap_days
        bad = np.array([bad_gap[i:i + C.WINDOW - 1].any() for i in idx])
        n_dropped += int(bad.sum())
        win = win[~bad]
        parts["stock"].append(np.full(len(win), sid))
        parts["dates"].append(dates[win])
        parts["close"].append(df["Close"].values[win])
        for col in ("open", "high", "low", "volume"):
            parts[col].append(df[col.capitalize()].values[win[:, :C.L_IN]])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    return WindowSet(**out, n_dropped=n_dropped)


# --------------------------------------------------------------------------
# Scaling constants (estimated on the training stocks only)
# --------------------------------------------------------------------------
@dataclass
class Scaler:
    s_ret: float   # std of daily log-returns (train)
    s_vol: float   # std of the within-week relative log-volume (train)
    mu_ret: float  # mean daily log-return (train) - used by the RW+drift baseline

    @classmethod
    def fit(cls, frames: dict[int, pd.DataFrame], windows: WindowSet) -> "Scaler":
        rets = []
        for df in frames.values():
            d = df["Date"].values.astype("datetime64[D]")
            ok = np.diff(d).astype(int) <= C.MAX_GAP_DAYS      # skip suspensions
            rets.append(np.diff(np.log(df["Close"].values))[ok])
        rets = np.concatenate(rets)
        v = _relative_log_volume(windows.volume)
        return cls(s_ret=float(rets.std()), s_vol=float(v.std()), mu_ret=float(rets.mean()))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        return cls(**json.loads(path.read_text()))


def _relative_log_volume(volume: np.ndarray) -> np.ndarray:
    lv = np.log1p(volume)
    return lv - lv.mean(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Features / targets for the DIRECT (multi-output) models
# --------------------------------------------------------------------------
def direct_features(w: WindowSet, sc: Scaler, feature_set: str) -> np.ndarray:
    """Input tensor (N, 5, F) built exclusively from the input week.

    close : F=1  -> z_t = log(C_t / C_5) / s
    ohlcv : F=5  -> [z_t, log(O_t/C_t)/s, log(H_t/C_t)/s, log(L_t/C_t)/s, v_t/s_vol]
            v_t = log(1+V_t) - mean_week log(1+V)   (relative trading activity)
    """
    c_in = w.close[:, :C.L_IN]
    z = np.log(c_in / w.last_close[:, None]) / sc.s_ret
    if feature_set == "close":
        feats = [z]
    elif feature_set == "ohlcv":
        feats = [z,
                 np.log(w.open / c_in) / sc.s_ret,
                 np.log(w.high / c_in) / sc.s_ret,
                 np.log(w.low / c_in) / sc.s_ret,
                 _relative_log_volume(w.volume) / sc.s_vol]
    else:
        raise ValueError(feature_set)
    return np.stack(feats, axis=-1).astype("float32")


def direct_targets(w: WindowSet, sc: Scaler) -> np.ndarray:
    """y_h = log(C_{5+h} / C_5) / s,  h = 1..5  -> (N, 5)."""
    return (np.log(w.target_close / w.last_close[:, None]) / sc.s_ret).astype("float32")


# --------------------------------------------------------------------------
# Sequences for the AUTOREGRESSIVE Vanilla RNN
# --------------------------------------------------------------------------
def log_returns(w: WindowSet, sc: Scaler) -> np.ndarray:
    """r_t = log(C_t / C_{t-1}) / s for t = 2..10  -> (N, 9)."""
    return (np.diff(np.log(w.close), axis=1) / sc.s_ret).astype("float32")


def ar_training_pairs(w: WindowSet, sc: Scaler) -> tuple[np.ndarray, np.ndarray]:
    """Teacher forcing: inputs r_2..r_9, one-step-ahead targets r_3..r_10."""
    r = log_returns(w, sc)
    return r[:, :-1, None], r[:, 1:, None]


def ar_context(w: WindowSet, sc: Scaler) -> np.ndarray:
    """What the AR model sees at forecast time: r_2..r_5 (4 returns of the input week)."""
    return log_returns(w, sc)[:, :C.L_IN - 1, None]


def returns_to_cumulative(r_hat: np.ndarray) -> np.ndarray:
    """Predicted returns r_6..r_10 -> cumulative log-ratios y_h = sum_{k<=h} r_{5+k}."""
    return np.cumsum(r_hat, axis=1)


# --------------------------------------------------------------------------
# Inverse transform
# --------------------------------------------------------------------------
def to_prices(w: WindowSet, y_scaled: np.ndarray, sc: Scaler) -> np.ndarray:
    """C_hat_{5+h} = C_5 * exp(s * y_hat_h)."""
    return w.last_close[:, None] * np.exp(sc.s_ret * np.asarray(y_scaled, dtype="float64"))


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------
def load_all() -> tuple[dict, dict, Scaler]:
    """Frames and windows of every split + scaler fitted on train."""
    frames = {s: load_split(s) for s in C.SPLITS}
    windows = {s: build_windows(frames[s]) for s in C.SPLITS}
    sc = Scaler.fit(frames["train"], windows["train"])
    return frames, windows, sc
