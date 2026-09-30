"""Generate every number and table of the report directly from results/.

Nothing in the report's tables is typed by hand: this script writes
report/generated/*.tex from the csv/json files produced by the pipeline, so the
report and the code can never disagree.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stockcast import config as C  # noqa: E402
from stockcast import data as D  # noqa: E402

GEN = ROOT / "report" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
T = C.TAB_DIR


def g(x, nd=2):
    """Number for math/table mode."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def gt(x, nd=2):
    """Number for text mode."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def pfmt(p):
    if p < 0.001:
        return "<0.001"
    return g(p, 3)


def ptext(p):
    if p < 0.001:
        return "<0.001"
    return gt(p, 3)


ENG = lambda s: r"\eng{" + s + "}"  # noqa: E731


def hp_compact(fam, hp):
    if fam in ("vrnn_ar", "vrnn_direct"):
        return f"layers={hp['layers']}, units={hp['units']}"
    if fam == "seq2seq_attn":
        return f"units={hp['units']}, dropout={hp['dropout']}"
    return f"d={hp['d_model']}, N={hp['layers']}, heads={hp['heads']}, dropout={hp['dropout']}"


def main():
    eda = json.loads((T / "eda_summary.json").read_text())
    bp = json.loads((C.RESULTS_DIR / "baseline_params.json").read_text())
    sel = json.loads((T / "selection.json").read_text())
    best = json.loads((T / "best_configs.json").read_text())
    tun = pd.read_csv(T / "tuning_summary.csv")
    log = json.loads((C.LOG_DIR / "final_training.json").read_text())
    ov = pd.read_csv(T / "test_overall.csv").set_index("model")
    tm = pd.read_csv(T / "test_metrics.csv")
    vm = pd.read_csv(T / "validation_metrics.csv")
    dm = pd.read_csv(T / "test_dm.csv").set_index("model")
    pst = pd.read_csv(T / "test_per_stock.csv")
    cw = pd.read_csv(T / "test_calendar_weeks.csv")
    beh = pd.read_csv(T / "test_forecast_behaviour.csv").set_index("model")
    eb = pd.read_csv(T / "test_exposure_bias.csv")
    frames, W, sc = D.load_all()
    P = sel["proposed"]

    # ------------------------------------------------------------ numbers
    num = {}
    num["Ndropped"] = str(W["train"].n_dropped)
    num["Ntrain"] = f"{len(W['train']):,}"
    num["Nval"] = f"{len(W['validation']):,}"
    num["Ntest"] = f"{len(W['test']):,}"
    num["NcalTest"] = str(int(W["test"].calendar_week_mask().sum()))
    num["Sret"] = g(sc.s_ret, 4)
    num["MuRetPct"] = gt(100 * sc.mu_ret, 3)
    num["MuRetAnnPct"] = gt(100 * (np.exp(252 * sc.mu_ret) - 1), 0)
    num["ADFlevelTrain"] = str(eda["adf_logprice_nonstationary_train"])
    num["ADFlevelAll"] = str(eda["all_stocks"]["adf_logprice_nonstationary"])
    num["ADFretAll"] = str(eda["all_stocks"]["adf_returns_stationary"])
    num["RhoOne"] = g(eda["mean_rho1_train"], 3)
    num["LBretTrain"] = str(eda["lb_returns_reject_train"])
    num["LBsqTrain"] = str(eda["lb_sq_returns_reject_train"])
    num["Kurt"] = gt(eda["pooled_train_ex_kurt"], 1)
    num["NmktDates"] = str(eda["n_market_dates"])
    num["WdP"] = g(eda["market_weekday_F_p"], 3)
    num["WdPhc"] = g(eda["market_weekday_F_p_hc3"], 3)
    num["WdPtrim"] = g(eda["market_weekday_F_p_trimmed_1pct"], 3)
    num["WdRejTrain"] = str(eda["weekday_reject_train"])
    num["SESalpha"] = g(bp["ses_alpha"], 2)
    num["PropName"] = ENG(P)

    def o(model, col, nd=2):
        return gt(float(ov.loc[model, col]), nd)

    for tag, model in [("Pers", "Persistence"), ("Drift", "RW+drift"), ("Ar", "AR(4)-OLS"),
                       ("Vrnn", "VRNN-AR"), ("Prop", P), ("Trend", "Linear trend (LS)"),
                       ("SES", "SES")]:
        num[f"{tag}MAPE"] = o(model, "MAPE", 3)
        num[f"{tag}RMSElog"] = o(model, "RMSE_log", 3)
        num[f"{tag}U"] = o(model, "U", 4)
        num[f"{tag}DA"] = o(model, "DA", 1) if not np.isnan(ov.loc[model, "DA"]) else "—"
        num[f"{tag}MAE"] = o(model, "MAE", 4)
        num[f"{tag}Gain"] = gt(100 * (1 - float(ov.loc[model, "U"]) ** 2), 1)   # % MSE reduction vs persistence
    for tag, model in [("Prop", P), ("Vrnn", "VRNN-AR"), ("Drift", "RW+drift"), ("Ar", "AR(4)-OLS")]:
        if model != "Persistence":
            num[f"{tag}DMpers"] = g(float(dm.loc[model, "DM_vs_Persistence"]), 2)
            num[f"{tag}Ppers"] = ptext(float(dm.loc[model, "p_vs_Persistence"]))
    num["PropDMvrnn"] = g(float(dm.loc[P, "DM_vs_VRNN-AR"]), 2)
    num["PropPvrnn"] = ptext(float(dm.loc[P, "p_vs_VRNN-AR"]))
    for m_, tag in [("RW+drift", "Drift"), ("AR(4)-OLS", "Ar")]:
        num[f"{tag}DMprop"] = g(float(dm.loc[m_, f"DM_vs_{P}"]), 2)
        num[f"{tag}Pprop"] = ptext(float(dm.loc[m_, f"p_vs_{P}"]))
    vall = vm[vm.horizon == "all"].set_index("model")
    num["ValPersMSE"] = gt(float(vall.loc["Persistence", "MSE_scaled"]), 4)
    num["ValPropMSE"] = gt(float(vall.loc[P, "MSE_scaled"]), 4)
    num["ValVrnnMSE"] = gt(float(vall.loc["VRNN-AR", "MSE_scaled"]), 4)
    num["ValPropGain"] = gt(100 * (1 - float(vall.loc[P, "MSE_scaled"]) / float(vall.loc["Persistence", "MSE_scaled"])), 1)
    num["ValVrnnGain"] = gt(100 * (1 - float(vall.loc["VRNN-AR", "MSE_scaled"]) / float(vall.loc["Persistence", "MSE_scaled"])), 1)
    num["ActualSdFive"] = gt(float(beh.loc["ACTUAL", "sd_pred_5d_pct"]), 2)
    num["ActualMeanFive"] = gt(float(beh.loc["ACTUAL", "mean_pred_5d_pct"]), 2)
    num["PropSdFive"] = gt(float(beh.loc[P, "sd_pred_5d_pct"]), 2)
    num["PropMeanFive"] = gt(float(beh.loc[P, "mean_pred_5d_pct"]), 2)
    num["PropCorrFive"] = gt(float(beh.loc[P, "corr_pred_actual_5d"]), 3)
    num["VrnnSdFive"] = gt(float(beh.loc["VRNN-AR", "sd_pred_5d_pct"]), 2)
    num["VrnnMeanFive"] = gt(float(beh.loc["VRNN-AR", "mean_pred_5d_pct"]), 2)
    num["VrnnCorrFive"] = gt(float(beh.loc["VRNN-AR", "corr_pred_actual_5d"]), 3)
    num["DriftMeanFive"] = gt(float(beh.loc["RW+drift", "mean_pred_5d_pct"]), 2)
    num["EbClFive"] = gt(float(eb.mse_closed_loop_rel.iloc[-1]), 3)
    num["EbTfFive"] = gt(float(eb.mse_teacher_rel.iloc[-1]), 3)
    num["EbClOne"] = gt(float(eb.mse_closed_loop_rel.iloc[0]), 3)
    # seeds
    sd = ov[["seed_MAPE_mean", "seed_MAPE_std"]]
    num["PropSeedMAPE"] = gt(float(sd.loc[P, "seed_MAPE_mean"]), 3)
    num["PropSeedMAPEsd"] = gt(float(sd.loc[P, "seed_MAPE_std"]), 3)
    num["VrnnSeedMAPE"] = gt(float(sd.loc["VRNN-AR", "seed_MAPE_mean"]), 3)
    num["VrnnSeedMAPEsd"] = gt(float(sd.loc["VRNN-AR", "seed_MAPE_std"]), 3)
    # calendar weeks
    cwa = cw[cw.horizon == "all"].set_index("model")
    for tag, model in [("Pers", "Persistence"), ("Vrnn", "VRNN-AR"), ("Prop", P), ("Drift", "RW+drift")]:
        num[f"Cw{tag}MAPE"] = gt(float(cwa.loc[model, "MAPE"]), 3)
        num[f"Cw{tag}U"] = gt(float(cwa.loc[model, "U"]), 4)
    cwdm = pd.read_csv(T / "test_calendar_weeks_dm.csv").set_index("model")
    num["CwPropP"] = ptext(float(cwdm.loc[P, "p_vs_Persistence"]))
    num["CwVrnnP"] = ptext(float(cwdm.loc["VRNN-AR", "p_vs_Persistence"]))
    num["CwWeeks"] = str(int(cwdm.n_weeks.iloc[0]))
    num["VrnnDMdrift"] = g(-float(dm.loc["RW+drift", "DM_vs_VRNN-AR"]), 2)
    num["VrnnPdrift"] = ptext(float(dm.loc["RW+drift", "p_vs_VRNN-AR"]))
    num["VrnnDMar"] = g(-float(dm.loc["AR(4)-OLS", "DM_vs_VRNN-AR"]), 2)
    num["VrnnPar"] = ptext(float(dm.loc["AR(4)-OLS", "p_vs_VRNN-AR"]))
    num["VrnnPpersHOne"] = ptext(float(dm.loc["VRNN-AR", "p_h1_vs_Persistence"]))
    num["PropPpersHOne"] = ptext(float(dm.loc[P, "p_h1_vs_Persistence"]))
    num["PropPpersHFive"] = ptext(float(dm.loc[P, "p_h5_vs_Persistence"]))
    num["DirectPvrnn"] = ptext(float(dm.loc["VRNN-Direct [close]", "p_vs_VRNN-AR"]))
    for tag, model in [("DirClose", "VRNN-Direct [close]"), ("DirOhlcv", "VRNN-Direct [OHLCV]"),
                       ("SsClose", "Seq2Seq-Attn [close]"), ("SsOhlcv", "Seq2Seq-Attn [OHLCV]"),
                       ("TrClose", "Transformer [close]"), ("TrOhlcv", "Transformer [OHLCV]")]:
        num[f"{tag}U"] = o(model, "U", 4)
        num[f"{tag}DA"] = o(model, "DA", 1)
    import numpy as _np
    A = _np.load(T / f"attention_{sel['proposed_key'].replace('|', '_')}.npy")
    num["AttLastMin"] = gt(float(A[:, -1].min()), 2)
    num["AttLastMax"] = gt(float(A[:, -1].max()), 2)
    num["AttPrev"] = gt(float(A[:, -2].mean()), 2)
    num["AttFirstTwo"] = gt(float(A[:, :2].mean()), 2)
    At = _np.load(T / "attention_transformer_close.npy")
    num["AttTrLast"] = gt(float(At[:, -1].mean()), 2)

    # per-stock wins
    wide = pst.pivot(index="stock", columns="model", values="MAPE")
    num["PropBeatsPersStocks"] = str(int((wide[P] < wide["Persistence"]).sum()))
    num["PropBeatsVrnnStocks"] = str(int((wide[P] < wide["VRNN-AR"]).sum()))
    num["NtestStocks"] = str(len(wide))
    # h-specific
    th = tm.set_index(["model", "horizon"])
    for h, hw in ((1, "One"), (5, "Five")):
        num[f"PersMAPEh{hw}"] = gt(float(th.loc[("Persistence", f"h{h}"), "MAPE"]), 2)
        num[f"PropUh{hw}"] = gt(float(th.loc[(P, f"h{h}"), "U"]), 4)
        num[f"VrnnUh{hw}"] = gt(float(th.loc[("VRNN-AR", f"h{h}"), "U"]), 4)
    # epochs
    for tag, key in [("Prop", sel["proposed_key"]), ("Vrnn", "vrnn_ar|close")]:
        eps = [r["best_epoch"] for r in log[key]["runs"]]
        num[f"{tag}BestEpochs"] = ", ".join(str(e) for e in eps)
        num[f"{tag}Params"] = f"{log[key]['runs'][0]['n_params']:,}"

    assert all(k.isalpha() for k in num), [k for k in num if not k.isalpha()]
    with open(GEN / "numbers.tex", "w") as f:
        for k, v in num.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    (GEN / "numbers.json").write_text(json.dumps(num, indent=1, ensure_ascii=False))

    # ------------------------------------------------------------ configs table
    names = {"vrnn_ar": "VRNN-AR", "vrnn_direct": "VRNN-Direct",
             "seq2seq_attn": "Seq2Seq-Attn", "transformer": "Transformer"}
    rows = []
    key_order = ["vrnn_ar|close", "vrnn_direct|close", "vrnn_direct|ohlcv", "seq2seq_attn|close",
                 "seq2seq_attn|ohlcv", "transformer|close", "transformer|ohlcv"]
    for key in key_order:
        hp = best[key]
        fam, fs = key.split("|")
        cfg = tun[(tun.family == fam) & (tun.feature_set == fs)]
        r0 = cfg.iloc[0]
        hp_s = hp_compact(fam, hp)
        eps = [r["best_epoch"] for r in log[key]["runs"]]
        feat = "returns" if fam == "vrnn_ar" else fs
        hp_tex = hp_s.replace("_", "\\_")
        params = f"{int(r0.n_params):,}"
        rows.append(ENG(names[fam]) + " & " + ENG(feat) + " & " + ENG(hp_tex) + " & " + params
                    + " & " + str(len(cfg)) + " & " + g(r0.val_mse, 4) + " & " + g(np.mean(eps), 1)
                    + " \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
\centering\small
\caption{Selected hyper-parameters per model family and feature set (grid $\times$ 2 seeds; criterion: validation forecast MSE). $d$: model dimension, $N$: number of blocks. ``Epoch'': mean best-validation epoch over the 5 final training runs.}
\label{tab:configs}
\begin{tabular}{llllrrr}
\toprule
Model & Input & Selected configuration & Params & Configs & MSE$_{\text{val}}$ & Epoch\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "configs.tex").write_text(tex)

    # ------------------------------------------------------------ main test table
    order = ["Persistence", "RW+drift", "Linear trend (LS)", "SES", "AR(4)-OLS", "VRNN-AR",
             "VRNN-Direct [close]", "VRNN-Direct [OHLCV]", "Seq2Seq-Attn [close]",
             "Seq2Seq-Attn [OHLCV]", "Transformer [close]", "Transformer [OHLCV]"]
    cols = ["MAPE", "RMSE_log", "U", "MAE", "RMSE", "DA"]
    nd = {"MAPE": 3, "RMSE_log": 3, "U": 4, "MAE": 4, "RMSE": 4, "DA": 1}
    best_val = {c: (ov.loc[order, c].max() if c == "DA" else ov.loc[order, c].min()) for c in cols}
    lines = []
    for i, mname in enumerate(order):
        cells = []
        for c in cols:
            v = float(ov.loc[mname, c])
            s = g(v, nd[c])
            if not np.isnan(v) and np.isclose(v, best_val[c]):
                s = r"\textbf{" + s + "}"
            cells.append(s)
        if not np.isnan(ov.loc[mname, "seed_MAPE_mean"]):
            seed = g(ov.loc[mname, "seed_MAPE_mean"], 3) + r"\,$\pm$\," + g(ov.loc[mname, "seed_MAPE_std"], 3)
        else:
            seed = "—"
        label = ENG(mname) + (r"$^{\star}$" if mname == P else "")
        pre = r"\rowcolor{best}" if mname == P else ""
        lines.append(pre + label + " & " + " & ".join(cells) + " & " + seed + r" \\")
        if mname == "AR(4)-OLS":
            lines.append(r"\midrule")
    tex = r"""\begin{table}[t]
\centering\small
\caption{Test set (5 stocks, \Ntest{} windows): forecast errors pooled over all horizons ($h=1..5$). Neural models are 5-seed ensembles; the last column gives the MAPE of the individual seeds (mean $\pm$ s.d.). Bold: best value per column; $^{\star}$: proposed model (selected on the validation set).}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
Model & MAPE (\%) & RMSE$_{\log}$ (\%) & Theil $U$ & MAE (€) & RMSE (€) & DA (\%) & MAPE, seeds\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "table_main.tex").write_text(tex)

    # ------------------------------------------------------------ per-horizon table
    key_models = ["Persistence", "RW+drift", "AR(4)-OLS", "VRNN-AR", "VRNN-Direct [OHLCV]",
                  "Seq2Seq-Attn [close]", "Seq2Seq-Attn [OHLCV]", "Transformer [close]", "Transformer [OHLCV]"]
    lines = []
    for mname in key_models:
        d = tm[(tm.model == mname)].set_index("horizon")
        mape = " & ".join(g(d.loc[f"h{h}", "MAPE"], 2) for h in range(1, 6))
        u = " & ".join(("—" if mname == "Persistence" else g(d.loc[f"h{h}", "U"], 3)) for h in range(1, 6))
        pre = r"\rowcolor{best}" if mname == P else ""
        lines.append(pre + ENG(mname) + (r"$^{\star}$" if mname == P else "") + " & " + mape + " & " + u + r" \\")
    tex = r"""\begin{table}[t]
\centering\small
\caption{Test set: MAPE (\%) and Theil's $U$ per forecast horizon $h$ (sessions).}
\label{tab:horizon}
\setlength{\tabcolsep}{3.6pt}
\begin{tabular}{l*{5}{r}|*{5}{r}}
\toprule
& \multicolumn{5}{c|}{MAPE (\%)} & \multicolumn{5}{c}{Theil $U$}\\
Model & $h{=}1$ & 2 & 3 & 4 & 5 & $h{=}1$ & 2 & 3 & 4 & 5\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "table_horizon.tex").write_text(tex)

    # ------------------------------------------------------------ DM table
    lines = []
    for mname in order:
        if mname == "Persistence":
            continue
        r = dm.loc[mname]

        def cell(stat_col, p_col):
            if stat_col not in r or pd.isna(r.get(stat_col)):
                return "—", "—"
            st, p = float(r[stat_col]), float(r[p_col])
            s = g(st, 2)
            if p < 0.05:
                s = r"\textbf{" + s + "}"
            return s, pfmt(p)
        c1 = cell("DM_vs_Persistence", "p_vs_Persistence")
        c2 = cell("DM_h1_vs_Persistence", "p_h1_vs_Persistence")
        c3 = cell("DM_h5_vs_Persistence", "p_h5_vs_Persistence")
        c4 = cell("DM_vs_VRNN-AR", "p_vs_VRNN-AR")
        c5 = cell(f"DM_vs_{P}", f"p_vs_{P}")
        pre = r"\rowcolor{best}" if mname == P else ""
        lines.append(pre + ENG(mname) + (r"$^{\star}$" if mname == P else "") + " & " +
                     " & ".join(f"{a} & {b}" for a, b in (c1, c2, c3, c4, c5)) + r" \\")
    tex = r"""\begin{table}[t]
\centering\small
\caption{Diebold--Mariano tests on the test set (HLN correction, Newey--West with 9 lags, losses averaged across stocks per forecast date). A negative statistic means that the row model is more accurate than the reference model of the column. Bold: $p<0.05$.}
\label{tab:dm}
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{l*{10}{r}}
\toprule
& \multicolumn{6}{c}{vs.\ Persistence} & \multicolumn{2}{c}{vs.\ VRNN-AR} & \multicolumn{2}{c}{vs.\ \eng{""" + P + r"""}}\\
\cmidrule(lr){2-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
& \multicolumn{2}{c}{all $h$} & \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{all $h$} & \multicolumn{2}{c}{all $h$}\\
Model & DM & $p$ & DM & $p$ & DM & $p$ & DM & $p$ & DM & $p$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "table_dm.tex").write_text(tex)

    # ------------------------------------------------------------ per stock + calendar
    km = ["Persistence", "RW+drift", "AR(4)-OLS", "VRNN-AR", P]
    wide = pst.pivot(index="stock", columns="model", values="MAPE")[km]
    wide_u = pst.pivot(index="stock", columns="model", values="RMSE_log")[km]
    lines = []
    for sid in wide.index:
        best_m = wide.loc[sid].min()
        cells = []
        for m_ in km:
            s = g(wide.loc[sid, m_], 3)
            if np.isclose(wide.loc[sid, m_], best_m):
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"\\#{sid} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cwa = cw[cw.horizon == "all"].set_index("model")
    cells = []
    bm = min(cwa.loc[m_, "MAPE"] for m_ in km)
    for m_ in km:
        s = g(cwa.loc[m_, "MAPE"], 3)
        if np.isclose(cwa.loc[m_, "MAPE"], bm):
            s = r"\textbf{" + s + "}"
        cells.append(s)
    lines.append(r"\makecell[l]{Calendar\\ weeks (" + str(int(W['test'].calendar_week_mask().sum())) + ")} & " + " & ".join(cells) + r" \\")
    lines.append(r"\quad \eng{Theil} $U$ & " + " & ".join(
        ("—" if m_ == "Persistence" else g(cwa.loc[m_, "U"], 4)) for m_ in km) + r" \\")
    head = " & ".join(ENG(m_) + (r"$^{\star}$" if m_ == P else "") for m_ in km)
    tex = r"""\begin{table}[t]
\centering\small
\caption{Robustness checks on the test set: MAPE (\%) per stock and under the strict calendar-week protocol (input Monday--Friday, target the following Monday--Friday).}
\label{tab:robust}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l*{5}{r}}
\toprule
Stock & """ + head + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "table_robust.tex").write_text(tex)

    # ------------------------------------------------------------ validation table (appendix)
    lines = []
    for mname in order:
        r = vall.loc[mname]
        pre = r"\rowcolor{best}" if mname == P else ""
        lines.append(pre + ENG(mname) + (r"$^{\star}$" if mname == P else "") + " & " + g(r.MSE_scaled, 4) + " & " +
                     g(r.MAPE, 3) + " & " + g(r.U, 4) + r" \\")
    tex = r"""\begin{table}[h]
\centering\small
\caption{Validation set (3 stocks, \Nval{} windows). The MSE (in the scaled space) is the selection criterion of the proposed model.}
\label{tab:val}
\begin{tabular}{lrrr}
\toprule
Model & MSE$_{\text{val}}$ & MAPE (\%) & Theil $U$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "table_val.tex").write_text(tex)

    # ------------------------------------------------------------ full tuning table (appendix)
    lines = []
    for r in tun.itertuples():
        hp = json.loads(r.config)
        hp_s = hp_compact(r.family, hp)
        is_best = best[f"{r.family}|{r.feature_set}"] == hp
        pre = r"\rowcolor{best}" if is_best else ""
        lines.append(pre + ENG(names[r.family]) + " & " + ENG(r.feature_set) + " & " +
                     ENG(hp_s.replace("_", r"\_")) + " & " + g(r.val_mse, 4) + " & " + g(r.val_mse_sd, 4) +
                     " & " + g(r.best_epoch, 1) + r" \\")
    tex = r"""{\small
\setlength{\tabcolsep}{4pt}
\begin{longtable}{lllrrr}
\caption{Full hyper-parameter search results (mean and standard deviation over 2 seeds). Selected configurations are highlighted.}\label{tab:tuning}\\
\toprule
Model & Input & Configuration & MSE$_{\text{val}}$ & s.d. & Epoch\\
\midrule
\endfirsthead
\toprule
Model & Input & Configuration & MSE$_{\text{val}}$ & s.d. & Epoch\\
\midrule
\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}}
"""
    (GEN / "table_tuning.tex").write_text(tex)
    print("generated:", sorted(p.name for p in GEN.iterdir()))


if __name__ == "__main__":
    main()
