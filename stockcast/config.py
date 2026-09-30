"""Global configuration: paths, problem definition, seeds and hyper-parameter grids.

Every script imports its constants from here, so that the whole experimental
protocol is defined in a single place.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
PRED_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "models"
LOG_DIR = RESULTS_DIR / "logs"

SPLITS = ("train", "validation", "test")

# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
# One "week" = 5 consecutive trading sessions of the Athens Stock Exchange.
L_IN = 5    # input length  (sessions t-4 ... t)
H_OUT = 5   # output length (sessions t+1 ... t+5)
WINDOW = L_IN + H_OUT

# A window is discarded when two consecutive sessions inside it are more than
# MAX_GAP_DAYS calendar days apart. The longest holiday gap in the common
# calendar is 6 days (Easter), so only genuine trading suspensions are removed
# (e.g. stock #17, 09/12/2025 - 07/01/2026).
MAX_GAP_DAYS = 7

# --------------------------------------------------------------------------
# Training protocol
# --------------------------------------------------------------------------
FINAL_SEEDS = (0, 1, 2, 3, 4)      # 5 independent runs per final model
TUNING_SEEDS = (100, 101)          # 2 runs per configuration during tuning
BATCH_SIZE = 128
MAX_EPOCHS = 150
PATIENCE = 15                      # early stopping on validation forecast MSE
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4                # AdamW

# --------------------------------------------------------------------------
# Hyper-parameter grids (selected on the validation stocks only)
# --------------------------------------------------------------------------
GRIDS = {
    # Baseline required by the assignment: autoregressive Vanilla RNN
    "vrnn_ar": [
        dict(units=u, layers=l, dropout=0.0)
        for u in (16, 32, 64) for l in (1, 2)
    ],
    # Ablation: same Vanilla RNN, but direct (non-autoregressive) multi-horizon output
    "vrnn_direct": [
        dict(units=u, layers=l, dropout=0.0)
        for u in (16, 32, 64) for l in (1, 2)
    ],
    # Encoder-decoder RNN with Bahdanau (additive) attention (Lecture 2)
    "seq2seq_attn": [
        dict(units=u, dropout=d)
        for u in (16, 32, 64) for d in (0.0, 0.1)
    ],
    # Transformer encoder-decoder with learned horizon queries (Lecture 3)
    "transformer": [
        dict(d_model=d, heads=4, layers=l, ff_mult=2, dropout=0.1)
        for d in (16, 32) for l in (1, 2)
    ],
}

# Feature sets for the direct neural models
FEATURE_SETS = ("close", "ohlcv")
