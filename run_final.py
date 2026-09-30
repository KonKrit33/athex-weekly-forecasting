"""Final training: every model family with its selected configuration is trained
with 5 seeds; forecasts on the validation and test stocks are stored.

Also computes the classical baselines (their only free parameters - drift,
SES constant, AR coefficients - are estimated on the training stocks).

Outputs
  results/models/<family>__<features>__seed<k>.weights.h5
  results/predictions/<family>__<features>.npz   (val/test forecasts, all seeds)
  results/predictions/baselines.npz
  results/logs/final_training.json               (curves, epochs, timings)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json  # noqa: E402

import numpy as np  # noqa: E402

from stockcast import baselines as B  # noqa: E402
from stockcast import config as C  # noqa: E402
from stockcast import data as D  # noqa: E402
from stockcast import train as T  # noqa: E402
from run_tuning import EXPERIMENTS  # noqa: E402


def main() -> None:
    for d in (C.MODEL_DIR, C.PRED_DIR, C.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    frames, W, sc = D.load_all()
    sc.save(C.RESULTS_DIR / "scaler.json")
    best = json.loads((C.TAB_DIR / "best_configs.json").read_text())

    # ------------------------------------------------------------ baselines
    alpha = B.ses_fit(W["train"], sc)
    ar = B.AROLS().fit(W["train"], sc)
    base = {}
    for split in ("validation", "test"):
        w = W[split]
        base[f"Persistence|{split}"] = B.persistence(w, sc)
        base[f"RW+drift|{split}"] = B.rw_drift(w, sc)
        base[f"Linear trend (LS)|{split}"] = B.linear_trend(w, sc)
        base[f"SES|{split}"] = B.ses(w, sc, alpha)
        base[f"AR(4)-OLS|{split}"] = ar.forecast(w, sc)
    np.savez(C.PRED_DIR / "baselines.npz", **base)
    (C.RESULTS_DIR / "baseline_params.json").write_text(json.dumps(
        dict(ses_alpha=alpha, ar_coef=ar.coef.tolist(), drift_per_day=sc.mu_ret,
             s_ret=sc.s_ret), indent=2))

    # ------------------------------------------------------------ neural nets
    log = {}
    for fam, fs in EXPERIMENTS:
        key = f"{fam}|{fs}"
        hp = best[key]
        task = T.make_task(fam, fs, W, sc)
        X_val = T.forecast_inputs(fam, fs, W["validation"], sc)
        X_test = T.forecast_inputs(fam, fs, W["test"], sc)
        preds_val, preds_test, infos = [], [], []
        for seed in C.FINAL_SEEDS:
            model, info = T.fit(task, hp, seed)
            model.save_weights(C.MODEL_DIR / f"{fam}__{fs}__seed{seed}.weights.h5")
            preds_val.append(T.forecast(model, fam, X_val))
            preds_test.append(T.forecast(model, fam, X_test))
            infos.append(info)
            print(f"{key:22s} seed={seed} val_mse={info['best_val_mse']:.4f} "
                  f"epoch={info['best_epoch']} params={info['n_params']} ({info['seconds']:.0f}s)",
                  flush=True)
        np.savez(C.PRED_DIR / f"{fam}__{fs}.npz", val=np.stack(preds_val), test=np.stack(preds_test))
        log[key] = dict(hp=hp, runs=infos)
        (C.LOG_DIR / "final_training.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
