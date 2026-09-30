#!/usr/bin/env bash
# Reproduces every number and figure of the technical report (CPU, ~1.5 h in total).
set -euo pipefail
cd "$(dirname "$0")"

python -m pytest -q          # sanity tests (data pipeline, baselines, DM test, AR loop)
python run_eda.py            # EDA: stationarity, ACF, Ljung-Box, weekday seasonality
python run_tuning.py         # hyper-parameter selection on the validation stocks
python run_final.py          # 5 seeds per model, forecasts on validation / test
python run_evaluation.py     # selection of the proposed model, test metrics, DM tests, figures
