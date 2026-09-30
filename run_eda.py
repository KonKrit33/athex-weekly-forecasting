"""Exploratory data analysis (Lectures 4-5 "general approach", TSA notes).

1. plot the series (trend, sharp changes, outliers)
2. check stationarity of log-prices vs log-returns (ADF test)
3. sample ACF of returns and squared returns, Ljung-Box portmanteau test
4. weekly seasonality (period d = 5 sessions) via harmonic regression + F-test

Modelling decisions are based on the TRAINING stocks only; descriptive
statistics are reported for all stocks.

Outputs: results/figures/eda_*.pdf, results/tables/eda_*.csv, results/tables/eda_summary.json
"""

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller

from stockcast import config as C
from stockcast import plotstyle as ps
from stockcast.data import load_split

warnings.filterwarnings("ignore")
ps.apply()
plt = ps.plt


def clean_returns(df: pd.DataFrame) -> pd.Series:
    """Daily log-returns, excluding returns that span a trading suspension."""
    d = df["Date"].values.astype("datetime64[D]")
    r = np.diff(np.log(df["Close"].values))
    ok = np.diff(d).astype(int) <= C.MAX_GAP_DAYS
    return pd.Series(r[ok], index=df["Date"].iloc[1:][ok].values)


def harmonic_design(weekday: np.ndarray, k: int = 2, d: int = 5) -> np.ndarray:
    """Harmonic regression regressors cos/sin(2*pi*j*t/d), j = 1..k (Lecture 4-5)."""
    cols = []
    for j in range(1, k + 1):
        lam = 2 * np.pi * j / d
        cols += [np.cos(lam * weekday), np.sin(lam * weekday)]
    return np.column_stack(cols)


def harmonic_ftest(r: pd.Series) -> tuple[float, float, np.ndarray]:
    wd = pd.DatetimeIndex(r.index).dayofweek.values
    X = sm.add_constant(harmonic_design(wd))
    fit = sm.OLS(r.values, X).fit()
    return float(fit.fvalue), float(fit.f_pvalue), fit.params


def main() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    frames = {s: load_split(s) for s in C.SPLITS}

    # ------------------------------------------------------------------ table
    rows = []
    for split, fr in frames.items():
        for sid, df in fr.items():
            r = clean_returns(df)
            lp = np.log(df["Close"].values)
            lb = acorr_ljungbox(r.values, lags=[10])
            lb2 = acorr_ljungbox(r.values ** 2, lags=[10])
            _, f_p, _ = harmonic_ftest(r)
            rows.append(dict(
                split=split, stock=sid, sessions=len(df),
                start=df["Date"].iloc[0].date(), end=df["Date"].iloc[-1].date(),
                close_min=df["Close"].min(), close_max=df["Close"].max(),
                total_return_pct=100 * (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1),
                ann_vol_pct=100 * r.std() * np.sqrt(252),
                skew=stats.skew(r.values), ex_kurt=stats.kurtosis(r.values),
                rho1=acf(r.values, nlags=1, fft=True)[1],
                adf_p_logprice=adfuller(lp, regression="ct", autolag="AIC")[1],
                adf_p_returns=adfuller(r.values, regression="c", autolag="AIC")[1],
                lb10_p_returns=float(lb["lb_pvalue"].iloc[0]),
                lb10_p_sq_returns=float(lb2["lb_pvalue"].iloc[0]),
                weekday_F_p=f_p,
            ))
    tab = pd.DataFrame(rows)
    tab.to_csv(C.TAB_DIR / "eda_per_stock.csv", index=False, float_format="%.4f")

    # --------------------------------------------------- pooled (train only)
    # The 17 training stocks share the same dates, so their returns are strongly
    # cross-correlated (market factor): pooling them as independent observations
    # would grossly overstate significance. We therefore test the weekday effect
    # on the equal-weighted cross-sectional mean return (one value per date).
    tr = tab[tab.split == "train"]
    R = pd.concat({k: clean_returns(df) for k, df in frames["train"].items()}, axis=1)
    r_train = pd.concat([clean_returns(df) for df in frames["train"].values()])
    mkt = R.mean(axis=1).dropna()
    F, Fp, params = harmonic_ftest(mkt)
    wd = pd.DatetimeIndex(mkt.index).dayofweek.values
    Xh = sm.add_constant(harmonic_design(wd))
    Fp_hc3 = float(sm.OLS(mkt.values, Xh).fit(cov_type="HC3").f_test(np.eye(5)[1:]).pvalue)
    q_lo, q_hi = mkt.quantile([0.01, 0.99])
    mkt_trim = mkt[(mkt > q_lo) & (mkt < q_hi)]
    _, Fp_trim, _ = harmonic_ftest(mkt_trim)
    by_wd = pd.DataFrame({"r": mkt.values, "wd": wd}).groupby("wd")["r"].agg(["mean", "median", "std", "count"])
    extreme_days = {str(k.date()): float(1e4 * v) for k, v in
                    mkt.reindex(mkt.abs().sort_values(ascending=False).index[:6]).items()}
    summary = dict(
        n_train_returns=int(len(r_train)),
        pooled_train_mean_ret=float(r_train.mean()),
        pooled_train_std_ret=float(r_train.std()),
        pooled_train_ex_kurt=float(stats.kurtosis(r_train.values)),
        jarque_bera_p=float(stats.jarque_bera(r_train.values).pvalue),
        adf_logprice_nonstationary_train=int((tr.adf_p_logprice > 0.05).sum()),
        adf_returns_stationary_train=int((tr.adf_p_returns < 0.05).sum()),
        lb_returns_reject_train=int((tr.lb10_p_returns < 0.05).sum()),
        lb_sq_returns_reject_train=int((tr.lb10_p_sq_returns < 0.05).sum()),
        weekday_reject_train=int((tr.weekday_F_p < 0.05).sum()),
        n_train_stocks=int(len(tr)),
        market_weekday_F=F, market_weekday_F_p=Fp, market_weekday_F_p_hc3=Fp_hc3,
        market_weekday_F_p_trimmed_1pct=float(Fp_trim), n_market_dates=int(len(mkt)),
        market_extreme_days_bp=extreme_days,
        weekday_median_ret_bp={int(k): float(1e4 * v) for k, v in by_wd["median"].items()},
        mean_rho1_train=float(tr.rho1.mean()),
        weekday_mean_ret_bp={int(k): float(1e4 * v) for k, v in by_wd["mean"].items()},
        all_stocks=dict(
            adf_logprice_nonstationary=int((tab.adf_p_logprice > 0.05).sum()),
            adf_returns_stationary=int((tab.adf_p_returns < 0.05).sum()),
            lb_returns_reject=int((tab.lb10_p_returns < 0.05).sum()),
            lb_sq_returns_reject=int((tab.lb10_p_sq_returns < 0.05).sum()),
            weekday_reject=int((tab.weekday_F_p < 0.05).sum()),
            n=int(len(tab))),
    )
    (C.TAB_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------------------------------------------------- figure: paths
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for split, fr in frames.items():
        for i, (sid, df) in enumerate(fr.items()):
            y = np.log(df["Close"] / df["Close"].iloc[0])
            ax.plot(df["Date"], y, color=ps.SPLIT_COLORS[split], lw=0.9, alpha=0.85,
                    label=f"{split} ({len(fr)} stocks)" if i == 0 else None)
    ax.axhline(0, color=ps.INK2, lw=0.8)
    ax.set_ylabel(r"$\log(C_t / C_{\mathrm{first}})$")
    ax.legend(loc="upper left")
    fig.savefig(C.FIG_DIR / "eda_paths.pdf")
    plt.close(fig)

    # ----------------------------------------------------------- figure: ACF
    nl = 20
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, sq, title in [(axes[0], False, r"(a) ACF of log-returns $r_t$"),
                          (axes[1], True, r"(b) ACF of squared returns $r_t^2$")]:
        acfs, ns = [], []
        for df in frames["train"].values():
            r = clean_returns(df).values
            x = r ** 2 if sq else r
            acfs.append(acf(x, nlags=nl, fft=True)[1:])
            ns.append(len(x))
        acfs = np.array(acfs)
        lags = np.arange(1, nl + 1)
        for a in acfs:
            ax.plot(lags, a, color=ps.BLUE, alpha=0.18, lw=0.8)
        ax.plot(lags, acfs.mean(0), color=ps.BLUE, lw=2, marker="o", ms=3,
                label="mean of the 17 training stocks")
        band = 1.96 / np.sqrt(np.median(ns))
        ax.axhspan(-band, band, color=ps.GRID, alpha=0.8, lw=0, label=r"$\pm 1.96/\sqrt{n}$")
        ax.axhline(0, color=ps.INK2, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("lag (sessions)")
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[0].set_ylabel("sample autocorrelation")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.savefig(C.FIG_DIR / "eda_acf.pdf")
    plt.close(fig)

    # -------------------------------------------- figure: distribution + weekday
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.35))
    z = (r_train - r_train.mean()) / r_train.std()
    bins = np.linspace(-8, 8, 81)
    axes[0].hist(z, bins=bins, density=True, color=ps.BLUE, alpha=0.85, label="train (standardised)")
    xx = np.linspace(-8, 8, 400)
    axes[0].plot(xx, stats.norm.pdf(xx), color=ps.ORANGE, lw=1.6, label="N(0,1)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-5, 1)
    axes[0].set_xlabel("standardised daily log-return")
    axes[0].set_title(f"(a) Return distribution (excess kurtosis {summary['pooled_train_ex_kurt']:.1f})")
    axes[0].legend(loc="upper left", fontsize=7)

    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    m = 1e4 * by_wd["mean"].values
    se = 1e4 * (by_wd["std"] / np.sqrt(by_wd["count"])).values
    axes[1].bar(range(5), m, yerr=1.96 * se, color=ps.BLUE, width=0.6,
                error_kw=dict(ecolor=ps.INK2, lw=1, capsize=3), label="mean return ± 95% CI")
    axes[1].plot(range(5), 1e4 * by_wd["median"].values, ls="none", marker="D", ms=4,
                 color=ps.INK, label="median")
    tt = np.linspace(-0.4, 4.4, 200)
    fit_curve = 1e4 * (params[0] + harmonic_design(tt) @ params[1:])
    axes[1].plot(tt, fit_curve, color=ps.ORANGE, lw=1.6, label="harmonic regression (d=5, k=2)")
    axes[1].axhline(0, color=ps.INK2, lw=0.8)
    axes[1].set_xticks(range(5), names)
    axes[1].set_ylabel("basis points (0.01%)")
    axes[1].set_title(f"(b) Day-of-week effect, F-test p = {Fp:.3f}")
    axes[1].set_ylim(-45, 110)
    axes[1].legend(loc="upper left", fontsize=7)
    fig.savefig(C.FIG_DIR / "eda_dist_weekday.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
