"""Training utilities shared by tuning and final training.

Model selection / early stopping criterion (identical for ALL neural models):
the mean squared error of the 5-step-ahead forecast on the validation stocks,
    MSE_val = mean_{n,h} (y_hat_{n,h} - y_{n,h})^2,
computed with the model used exactly as at test time (closed loop for the
autoregressive RNN). This avoids selecting the AR model on its teacher-forced
one-step loss, which is not the quantity we care about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import keras
import tensorflow as tf

from . import config as C
from . import data as D
from .models import ar_closed_loop, build


@dataclass
class Task:
    """Arrays needed to train/evaluate one model family on one feature set."""
    family: str
    feature_set: str
    X_fit: np.ndarray          # training inputs
    Y_fit: np.ndarray          # training targets
    X_val_fc: np.ndarray       # validation inputs as used for forecasting
    Y_val: np.ndarray          # validation 5-step targets (N, 5)
    n_features: int


def make_task(family: str, feature_set: str, windows: dict, sc: D.Scaler) -> Task:
    tr, va = windows["train"], windows["validation"]
    if family == "vrnn_ar":
        X, Y = D.ar_training_pairs(tr, sc)               # teacher forcing pairs
        return Task(family, "close", X, Y, D.ar_context(va, sc), D.direct_targets(va, sc), 1)
    X = D.direct_features(tr, sc, feature_set)
    return Task(family, feature_set, X, D.direct_targets(tr, sc),
                D.direct_features(va, sc, feature_set), D.direct_targets(va, sc), X.shape[-1])


def forecast(model: keras.Model, family: str, X: np.ndarray) -> np.ndarray:
    """Scaled log-ratio forecasts (N, 5) for inputs prepared for that family."""
    if family == "vrnn_ar":
        return ar_closed_loop(model, X)
    return np.asarray(model.predict(X, batch_size=4096, verbose=0), dtype="float64")


def forecast_inputs(family: str, feature_set: str, w: D.WindowSet, sc: D.Scaler) -> np.ndarray:
    if family == "vrnn_ar":
        return D.ar_context(w, sc)
    return D.direct_features(w, sc, feature_set)


class ForecastEarlyStopping(keras.callbacks.Callback):
    """Early stopping on the validation *forecast* MSE; restores the best weights."""

    def __init__(self, family, X_val, Y_val, patience):
        super().__init__()
        self.family, self.X_val, self.Y_val, self.patience = family, X_val, Y_val, patience
        self.best, self.best_epoch, self.best_weights, self.wait = np.inf, -1, None, 0
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        mse = float(np.mean((forecast(self.model, self.family, self.X_val) - self.Y_val) ** 2))
        self.history.append(mse)
        if logs is not None:
            logs["val_forecast_mse"] = mse
        if mse < self.best - 1e-6:
            self.best, self.best_epoch, self.wait = mse, epoch, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def fit(task: Task, hp: dict, seed: int, max_epochs: int = C.MAX_EPOCHS,
        patience: int = C.PATIENCE, verbose: int = 0):
    """Build + train one model. Returns (model, info dict)."""
    keras.utils.set_random_seed(seed)          # python, numpy, tensorflow seeds
    tf.config.experimental.enable_op_determinism()
    model = build(task.family, task.n_features, hp)
    model.compile(optimizer=keras.optimizers.AdamW(C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY,
                                                   clipnorm=1.0),
                  loss="mse")
    es = ForecastEarlyStopping(task.family, task.X_val_fc, task.Y_val, patience)
    t0 = time.time()
    hist = model.fit(task.X_fit, task.Y_fit, batch_size=C.BATCH_SIZE, epochs=max_epochs,
                     shuffle=True, verbose=verbose, callbacks=[es])
    info = dict(best_val_mse=es.best, best_epoch=es.best_epoch + 1,
                epochs_run=len(hist.history["loss"]), seconds=time.time() - t0,
                train_loss=[float(v) for v in hist.history["loss"]],
                val_forecast_mse=es.history, n_params=int(model.count_params()))
    return model, info
