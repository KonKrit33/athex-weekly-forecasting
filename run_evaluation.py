"""Evaluation of all predictors.

Protocol
  1. The PROPOSED predictor is selected on the validation stocks only: the
     attention-based candidate (Seq2Seq-Attn or Transformer, close or OHLCV
     input) with the lowest validation MSE of its 5-seed ensemble.
  2. Only then are the test stocks scored. The final forecast of every neural
     model is the average of its 5 seeds (ensemble); the dispersion of the 5
     single-seed scores is reported as well.
  3. Metrics in price space (MAE, RMSE, MAPE), in log space (RMSE_log, Theil U
     vs persistence), directional accuracy, Diebold-Mariano tests, per-stock
     and per-horizon breakdowns, and a strict calendar-week protocol.

Outputs: results/tables/*.csv|json, results/figures/*.pdf
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stockcast import config as C  # noqa: E402
from stockcast import data as D  # noqa: E402
from stockcast import metrics as M  # noqa: E402
from stockcast import plotstyle as ps  # noqa: E402
from stockcast.models import build  # noqa: E402

ps.apply()
plt = ps.plt

NEURAL = {
    "vrnn_ar|close": "VRNN-AR",
    "vrnn_direct|close": "VRNN-Direct [close]",
    "vrnn_direct|ohlcv": "VRNN-Direct [OHLCV]",
    "seq2seq_attn|close": "Seq2Seq-Attn [close]",
    "seq2seq_attn|ohlcv": "Seq2Seq-Attn [OHLCV]",
    "transformer|close": "Transformer [close]",
    "transformer|ohlcv": "Transformer [OHLCV]",
}
CANDIDATES = ["seq2seq_attn|close", "seq2seq_attn|ohlcv", "transformer|close", "transformer|ohlcv"]
BASELINES = ["Persistence", "RW+drift", "Linear trend (LS)", "SES", "AR(4)-OLS"]


def load_predictions():
    base = np.load(C.PRED_DIR / "baselines.npz")
    ens, seeds = {}, {}
    for split in ("validation", "test"):
        for b in BASELINES:
            ens[(b, split)] = base[f"{b}|{split}"]
    for key, name in NEURAL.items():
        f = np.load(C.PRED_DIR / f"{key.replace('|', '__')}.npz")
        for split, arr in (("validation", f["val"]), ("test", f["test"])):
            seeds[(name, split)] = arr                 # (S, N, 5)
            ens[(name, split)] = arr.mean(0)           # ensemble
    return ens, seeds


def evaluate(ens, seeds, W, sc, split, mask=None):
    w = W[split] if mask is None else W[split].subset(mask)
    y = D.direct_targets(w, sc)
    res, seed_rows = {}, []
    for (name, sp), yhat in ens.items():
        if sp != split:
            continue
        yh = yhat if mask is None else yhat[mask]
        res[name] = M.price_metrics(D.to_prices(w, yh, sc), w.target_close, w.last_close)
        if (name, sp) in seeds:
            for k, ys in enumerate(seeds[(name, sp)]):
                ys = ys if mask is None else ys[mask]
                m = M.price_metrics(D.to_prices(w, ys, sc), w.target_close, w.last_close)["all"]
                seed_rows.append(dict(model=name, seed=k, **m,
                                      MSE_scaled=float(np.mean((ys - y) ** 2))))
    df = M.metrics_frame(res)
    df["MSE_scaled"] = df.apply(
        lambda r: float(np.mean(((ens[(r.model, split)] if mask is None else ens[(r.model, split)][mask])
                                 - y)[:, slice(None) if r.horizon == "all" else
                                      slice(int(r.horizon[1]) - 1, int(r.horizon[1]))] ** 2)),
        axis=1)
    return df, pd.DataFrame(seed_rows)


def main() -> None:
    frames, W, sc = D.load_all()
    ens, seeds = load_predictions()
    best = json.loads((C.TAB_DIR / "best_configs.json").read_text())

    # ================================================= 1. selection on VALIDATION
    val_df, val_seed = evaluate(ens, seeds, W, sc, "validation")
    val_df.to_csv(C.TAB_DIR / "validation_metrics.csv", index=False, float_format="%.5f")
    val_all = val_df[val_df.horizon == "all"].set_index("model")
    cand = {NEURAL[k]: float(val_all.loc[NEURAL[k], "MSE_scaled"]) for k in CANDIDATES}
    proposed_key = min(CANDIDATES, key=lambda k: cand[NEURAL[k]])
    PROPOSED = NEURAL[proposed_key]
    selection = dict(candidates_val_mse=cand, proposed=PROPOSED, proposed_key=proposed_key,
                     proposed_config=best[proposed_key])
    (C.TAB_DIR / "selection.json").write_text(json.dumps(selection, indent=2))
    print("Selected on validation:", PROPOSED, cand)

    # ================================================= 2. TEST
    wt = W["test"]
    yt = D.direct_targets(wt, sc)
    test_df, test_seed = evaluate(ens, seeds, W, sc, "test")
    test_df.to_csv(C.TAB_DIR / "test_metrics.csv", index=False, float_format="%.5f")
    test_seed.to_csv(C.TAB_DIR / "test_metrics_per_seed.csv", index=False, float_format="%.5f")
    val_seed.to_csv(C.TAB_DIR / "validation_metrics_per_seed.csv", index=False, float_format="%.5f")

    order = BASELINES + list(NEURAL.values())
    overall = (test_df[test_df.horizon == "all"].set_index("model").loc[order]
               [["MAE", "RMSE", "MAPE", "RMSE_log", "U", "DA", "MSE_scaled"]])
    sd = test_seed.groupby("model")[["MAPE", "RMSE_log"]].agg(["mean", "std"])
    sd.columns = [f"seed_{a}_{b}" for a, b in sd.columns]
    overall = overall.join(sd)
    overall.to_csv(C.TAB_DIR / "test_overall.csv", float_format="%.5f")

    # ---------------------------------------------------- Diebold-Mariano (test)
    origin = wt.origin
    dm_rows = []
    for name in order:
        la = M.log_sq_loss(ens[(name, "test")], yt)
        row = dict(model=name)
        for ref in ("Persistence", "VRNN-AR", PROPOSED):
            if name == ref:
                continue
            lb = M.log_sq_loss(ens[(ref, "test")], yt)
            dbar, stat, p = M.diebold_mariano(la, lb, origin)
            row[f"DM_vs_{ref}"] = stat
            row[f"p_vs_{ref}"] = p
        for hz in (1, 5):
            la_h = M.log_sq_loss(ens[(name, "test")], yt, hz)
            lb_h = M.log_sq_loss(ens[("Persistence", "test")], yt, hz)
            if name != "Persistence":
                _, stat, p = M.diebold_mariano(la_h, lb_h, origin)
                row[f"DM_h{hz}_vs_Persistence"], row[f"p_h{hz}_vs_Persistence"] = stat, p
        dm_rows.append(row)
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(C.TAB_DIR / "test_dm.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------ per stock (test)
    key_models = ["Persistence", "RW+drift", "AR(4)-OLS", "VRNN-AR", PROPOSED]
    ps_rows = []
    for sid in np.unique(wt.stock):
        msk = wt.stock == sid
        ws = wt.subset(msk)
        for name in key_models:
            m = M.price_metrics(D.to_prices(ws, ens[(name, "test")][msk], sc),
                                ws.target_close, ws.last_close)["all"]
            ps_rows.append(dict(stock=int(sid), model=name, **m))
    per_stock = pd.DataFrame(ps_rows)
    per_stock.to_csv(C.TAB_DIR / "test_per_stock.csv", index=False, float_format="%.5f")

    # --------------------------------------------- strict calendar-week protocol
    cw = wt.calendar_week_mask()
    cw_df, _ = evaluate(ens, {}, W, sc, "test", mask=cw)
    cw_df.to_csv(C.TAB_DIR / "test_calendar_weeks.csv", index=False, float_format="%.5f")
    cw_dm = []
    for name in ("RW+drift", "AR(4)-OLS", "VRNN-AR", PROPOSED):
        la = M.log_sq_loss(ens[(name, "test")][cw], yt[cw])
        lb = M.log_sq_loss(ens[("Persistence", "test")][cw], yt[cw])
        # calendar-week windows do not overlap: lag 0 (no HAC correction needed)
        _, stat, p = M.diebold_mariano(la, lb, wt.origin[cw], lag=0)
        cw_dm.append(dict(model=name, DM_vs_Persistence=stat, p_vs_Persistence=p,
                          n_windows=int(cw.sum()), n_weeks=int(len(np.unique(wt.origin[cw])))))
    pd.DataFrame(cw_dm).to_csv(C.TAB_DIR / "test_calendar_weeks_dm.csv", index=False, float_format="%.4f")

    # ---------------------------------------------- forecast behaviour summary
    beh = []
    for name in order:
        yh = ens[(name, "test")] * sc.s_ret
        beh.append(dict(model=name, mean_pred_5d_pct=100 * float(yh[:, 4].mean()),
                        sd_pred_5d_pct=100 * float(yh[:, 4].std()),
                        corr_pred_actual_5d=float(np.corrcoef(yh[:, 4], yt[:, 4])[0, 1])
                        if yh[:, 4].std() > 0 else np.nan))
    beh.append(dict(model="ACTUAL", mean_pred_5d_pct=100 * float((yt[:, 4] * sc.s_ret).mean()),
                    sd_pred_5d_pct=100 * float((yt[:, 4] * sc.s_ret).std()), corr_pred_actual_5d=1.0))
    pd.DataFrame(beh).to_csv(C.TAB_DIR / "test_forecast_behaviour.csv", index=False, float_format="%.4f")

    # ======================================================== FIGURES
    figures(W, sc, ens, test_df, PROPOSED, proposed_key, best, frames)
    print(overall.round(4).to_string())
    print(dm.round(3).to_string())


# ---------------------------------------------------------------------------
def figures(W, sc, ens, test_df, PROPOSED, proposed_key, best, frames):
    wt = W["test"]
    fs_prop = proposed_key.split("|")[1]
    tag = "[close]" if fs_prop == "close" else "[OHLCV]"
    show = ["Persistence", "RW+drift", "AR(4)-OLS", "VRNN-AR", f"VRNN-Direct {tag}",
            f"Transformer {tag}", f"Seq2Seq-Attn {tag}"]
    if PROPOSED not in show:
        show.append(PROPOSED)
    hz = [f"h{h}" for h in range(1, C.H_OUT + 1)]

    # --- F1: MAPE and Theil U per horizon
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.3))
    for name in show:
        d = test_df[(test_df.model == name) & (test_df.horizon.isin(hz))]
        lw = 2.4 if name == PROPOSED else 1.4
        ls = "--" if name in ("Persistence", "RW+drift") else "-"
        lab = f"{name} (proposed)" if name == PROPOSED else name
        axes[0].plot(range(1, 6), d.MAPE.values, marker="o", ms=3, lw=lw, ls=ls,
                     color=ps.model_color(name), label=lab)
        axes[1].plot(range(1, 6), d.U.values, marker="o", ms=3, lw=lw, ls=ls,
                     color=ps.model_color(name))
    axes[0].set_title("(a) MAPE per horizon (test)")
    axes[0].set_xlabel("horizon h (sessions)")
    axes[0].set_ylabel("MAPE (%)")
    axes[1].set_title("(b) Theil U = RMSE$_{log}$ / RMSE$_{log}$(persistence)")
    axes[1].set_xlabel("horizon h (sessions)")
    axes[1].axhline(1, color=ps.INK2, lw=0.8)
    for ax in axes:
        ax.set_xticks(range(1, 6))
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), fontsize=7)
    fig.savefig(C.FIG_DIR / "res_horizon.pdf")
    plt.close(fig)

    # --- F2: example forecasts on test stocks (non-overlapping weeks, last 12 weeks)
    sids = np.unique(wt.stock)
    fig, axes = plt.subplots(len(sids), 1, figsize=(7.2, 1.75 * len(sids)), sharex=True)
    for ax, sid in zip(axes, sids):
        idx = np.where(wt.stock == sid)[0]
        idx = idx[-60::5][-12:] if len(idx) >= 60 else idx[::5]
        df = frames["test"][sid]
        t0 = wt.dates[idx[0], 0]
        dd = df[df.Date >= pd.Timestamp(t0)]
        ax.plot(dd.Date, dd.Close, color=ps.INK, lw=1.1, label="actual close")
        for j, i in enumerate(idx):
            fut = pd.to_datetime(wt.dates[i, C.L_IN - 1:])
            for name, col in (("VRNN-AR", ps.model_color("VRNN-AR")), (PROPOSED, ps.model_color(PROPOSED))):
                p = D.to_prices(wt.subset(np.arange(len(wt)) == i), ens[(name, "test")][i:i + 1], sc)[0]
                ax.plot(fut, np.r_[wt.last_close[i], p], color=col, lw=1.4,
                        label=(name if j == 0 else None))
        ax.set_ylabel(f"#{sid} (€)")
    axes[0].legend(loc="upper left", ncol=3, fontsize=7)
    fig.autofmt_xdate()
    fig.savefig(C.FIG_DIR / "res_examples.pdf")
    plt.close(fig)

    # --- F3: attention maps (Seq2Seq-Attn and Transformer, OHLCV or proposed feature set)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.35))
    for ax, key in zip(axes, [f"seq2seq_attn|{fs_prop}", f"transformer|{fs_prop}"]):
        fam, fs = key.split("|")
        X = D.direct_features(wt, sc, fs)
        maps = []
        for seed in C.FINAL_SEEDS:
            m = build(fam, X.shape[-1], best[key])
            m.load_weights(C.MODEL_DIR / f"{fam}__{fs}__seed{seed}.weights.h5")
            if fam == "seq2seq_attn":
                _, a = m(X, return_attention=True)
                maps.append(np.asarray(a).mean(0))                        # (H, T)
            else:
                _, _, cross = m(X, return_attention=True)
                maps.append(np.asarray(cross[-1]).mean(axis=(0, 1)))      # heads & windows
        A = np.mean(maps, 0)
        np.save(C.TAB_DIR / f"attention_{fam}_{fs}.npy", A)
        im = ax.imshow(A, cmap="Blues", vmin=0.1, vmax=0.3, aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if A[i, j] > 0.26 else ps.INK)
        ax.set_xticks(range(5), [f"t−{4 - j}" if j < 4 else "t" for j in range(5)])
        ax.set_yticks(range(5), [f"h={i + 1}" for i in range(5)])
        ax.set_xlabel("day of the input week")
        ax.grid(False)
        ax.set_title(("(b) " if fam == "transformer" else "(a) ") + NEURAL[key]
                     + (": cross-attention" if fam == "transformer" else r": Bahdanau $\alpha_{ij}$"), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(C.FIG_DIR / "res_attention.pdf")
    plt.close(fig)

    # --- F4: exposure bias of the autoregressive Vanilla RNN
    fam, fs = "vrnn_ar", "close"
    r = D.log_returns(wt, sc)                        # r_2..r_10
    tf_pred, cl_pred = [], []
    for seed in C.FINAL_SEEDS:
        m = build(fam, 1, best["vrnn_ar|close"])
        m.load_weights(C.MODEL_DIR / f"{fam}__{fs}__seed{seed}.weights.h5")
        tf_pred.append(np.asarray(m(r[:, :-1, None]))[:, 3:, 0])   # predicts r_6..r_10 from TRUE past
    tf_pred = np.mean(tf_pred, 0)
    cum = ens[("VRNN-AR", "test")]
    cl_pred = np.diff(np.c_[np.zeros(len(cum)), cum], axis=1)          # closed-loop r_hat_6..r_hat_10
    r_true = r[:, 4:]
    mse_tf = ((tf_pred - r_true) ** 2).mean(0)
    mse_cl = ((cl_pred - r_true) ** 2).mean(0)
    mse_0 = (r_true ** 2).mean(0)
    y = D.direct_targets(wt, sc)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.3))
    k = np.arange(1, 6)
    axes[0].plot(k, mse_cl / mse_0, marker="o", color=ps.ORANGE, label="closed loop (actual use)")
    axes[0].plot(k, mse_tf / mse_0, marker="s", color=ps.VIOLET, ls="--",
                 label="teacher forcing (diagnostic)")
    axes[0].axhline(1, color=ps.INK2, lw=0.8)
    axes[0].set_title(r"(a) MSE($\hat r_{5+k}$) / MSE(0)", fontsize=9)
    axes[0].set_xlabel("step k")
    axes[0].legend(fontsize=7, loc="center right")
    sd_cl = 100 * sc.s_ret * cl_pred.std(0)
    sd_tf = 100 * sc.s_ret * tf_pred.std(0)
    axes[1].plot(k, sd_cl, marker="o", color=ps.ORANGE, label="closed loop")
    axes[1].plot(k, sd_tf, marker="s", ls="--", color=ps.VIOLET, label="teacher forcing")
    axes[1].set_ylim(0, None)
    axes[1].set_title(r"(b) std. dev. of forecasts $\hat r_{5+k}$ (%)", fontsize=9)
    axes[1].set_xlabel("step k")
    axes[1].legend(fontsize=7, loc="lower left")
    for ax in axes:
        ax.set_xticks(k)
    fig.savefig(C.FIG_DIR / "res_exposure_bias.pdf")
    plt.close(fig)
    pd.DataFrame(dict(k=k, mse_closed_loop_rel=mse_cl / mse_0, mse_teacher_rel=mse_tf / mse_0,
                      sd_closed_loop_pct=sd_cl, sd_teacher_pct=sd_tf)).to_csv(
        C.TAB_DIR / "test_exposure_bias.csv", index=False, float_format="%.5f")

    # --- F5: learning curves (seed 0) of the baseline and the proposed model
    log = json.loads((C.LOG_DIR / "final_training.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.3))
    for ax, key, name, lab in ((axes[0], "vrnn_ar|close", "VRNN-AR", "(a)"), (axes[1], proposed_key, PROPOSED, "(b)")):
        for s, run in enumerate(log[key]["runs"]):
            v = np.array(run["val_forecast_mse"])
            ax.plot(np.arange(1, len(v) + 1), v, color=ps.model_color(name), alpha=0.35 + 0.13 * s, lw=1)
            ax.plot(run["best_epoch"], run["best_val_mse"], "o", color=ps.model_color(name), ms=4)
        pers = float(np.mean(D.direct_targets(W["validation"], sc) ** 2))
        ax.axhline(pers, color=ps.NEUTRAL, ls="--", lw=1, label="persistence")
        ax.set_title(f"{lab} {name}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("validation forecast MSE")
    fig.savefig(C.FIG_DIR / "res_learning_curves.pdf")
    plt.close(fig)

    # --- F6: predicted vs realised 5-day log-return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    yh5 = 100 * ens[(PROPOSED, "test")][:, 4] * sc.s_ret
    y5 = 100 * y[:, 4] * sc.s_ret
    ax.scatter(yh5, y5, s=4, alpha=0.35, color=ps.BLUE, lw=0)
    lim = np.percentile(np.abs(y5), 99.5)
    ax.axhline(0, color=ps.INK2, lw=0.6)
    ax.axvline(0, color=ps.INK2, lw=0.6)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("forecast 5-day return (%)")
    ax.set_ylabel("realised 5-day return (%)")
    ax.set_title(f"{PROPOSED}, h = 5:  $\\rho$ = {np.corrcoef(yh5, y5)[0, 1]:.3f}", fontsize=9)
    fig.savefig(C.FIG_DIR / "res_pred_vs_actual.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
