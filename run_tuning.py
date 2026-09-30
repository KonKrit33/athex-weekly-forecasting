"""Hyper-parameter selection on the VALIDATION stocks (test stocks never touched).

For every model family / feature set / configuration of config.GRIDS the model
is trained with 2 seeds; the configuration with the lowest mean validation
forecast MSE is selected. Results are appended to results/tables/tuning.csv
(the script can be interrupted and resumed).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

from stockcast import config as C  # noqa: E402
from stockcast import data as D  # noqa: E402
from stockcast import train as T  # noqa: E402

EXPERIMENTS = [("vrnn_ar", "close")] + [
    (fam, fs) for fam in ("vrnn_direct", "seq2seq_attn", "transformer") for fs in C.FEATURE_SETS]


def main(only: str | None = None) -> None:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = C.TAB_DIR / "tuning.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(
        columns=["family", "feature_set", "config", "seed"])
    done_keys = set(zip(done.family, done.feature_set, done.config, done.seed))

    _, W, sc = D.load_all()
    for fam, fs in EXPERIMENTS:
        if only and fam != only:
            continue
        task = T.make_task(fam, fs, W, sc)
        for hp in C.GRIDS[fam]:
            key = json.dumps(hp, sort_keys=True)
            for seed in C.TUNING_SEEDS:
                if (fam, fs, key, seed) in done_keys:
                    continue
                _, info = T.fit(task, hp, seed)
                row = dict(family=fam, feature_set=fs, config=key, seed=seed,
                           val_mse=info["best_val_mse"], best_epoch=info["best_epoch"],
                           epochs=info["epochs_run"], seconds=round(info["seconds"], 1),
                           n_params=info["n_params"])
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{fam:13s} {fs:6s} {key:70s} seed={seed} val_mse={row['val_mse']:.4f} "
                      f"epoch={row['best_epoch']:3d} ({row['seconds']:.0f}s)", flush=True)

    # ---------------------------------------------------------------- select
    res = pd.read_csv(out)
    agg = (res.groupby(["family", "feature_set", "config"])
              .agg(val_mse=("val_mse", "mean"), val_mse_sd=("val_mse", "std"),
                   best_epoch=("best_epoch", "mean"), n_params=("n_params", "first"),
                   seconds=("seconds", "mean"))
              .reset_index().sort_values(["family", "feature_set", "val_mse"]))
    agg.to_csv(C.TAB_DIR / "tuning_summary.csv", index=False, float_format="%.5f")
    best = {f"{r.family}|{r.feature_set}": json.loads(r.config)
            for r in agg.groupby(["family", "feature_set"]).head(1).itertuples()}
    (C.TAB_DIR / "best_configs.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="restrict to one model family")
    main(ap.parse_args().only)
