"""Sanity tests of the data pipeline, the baselines and the evaluation code.

Run with:  python -m pytest -q
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest

from stockcast import baselines as B
from stockcast import config as C
from stockcast import data as D
from stockcast import metrics as M


@pytest.fixture(scope="module")
def setup():
    if not (C.DATA_DIR / "train").exists():
        pytest.skip("data/ folder not available")
    return D.load_all()


def test_splits_are_disjoint(setup):
    frames, _, _ = setup
    ids = [set(frames[s]) for s in C.SPLITS]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert [len(i) for i in ids] == [17, 3, 5]


def test_inputs_do_not_depend_on_target_week(setup):
    """Changing the future (target) prices must not change any model input."""
    _, W, sc = setup
    w = W["test"].subset(np.arange(len(W["test"])) < 50)
    w2 = w.subset(np.ones(len(w), bool))
    w2.close = w.close.copy()
    w2.close[:, C.L_IN:] *= 1.37
    for fs in C.FEATURE_SETS:
        np.testing.assert_array_equal(D.direct_features(w, sc, fs), D.direct_features(w2, sc, fs))
    np.testing.assert_array_equal(D.ar_context(w, sc), D.ar_context(w2, sc))


def test_inverse_transform_is_exact(setup):
    _, W, sc = setup
    w = W["validation"]
    np.testing.assert_allclose(D.to_prices(w, D.direct_targets(w, sc), sc), w.target_close, rtol=1e-6)


def test_ar_pairs_are_one_step_shifted(setup):
    _, W, sc = setup
    x, y = D.ar_training_pairs(W["train"], sc)
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
    np.testing.assert_allclose(D.returns_to_cumulative(D.log_returns(W["train"], sc)[:, C.L_IN - 1:]),
                               D.direct_targets(W["train"], sc), atol=1e-5)


def test_suspension_windows_removed(setup):
    _, W, _ = setup
    gaps = np.diff(W["train"].dates.astype("datetime64[D]").view("int64"), axis=1)
    assert gaps.max() <= C.MAX_GAP_DAYS
    assert W["train"].n_dropped == 9          # stock 17: 9 windows cross the 31-day gap


def test_calendar_week_mask(setup):
    _, W, _ = setup
    w = W["test"]
    m = w.calendar_week_mask()
    wd = pd.DatetimeIndex(w.dates[m].ravel()).dayofweek.values.reshape(-1, C.WINDOW)
    assert (wd == np.tile(np.arange(5), 2)).all()
    assert m.sum() > 0


def test_persistence_returns_last_close(setup):
    _, W, sc = setup
    w = W["test"]
    p = D.to_prices(w, B.persistence(w, sc), sc)
    np.testing.assert_allclose(p, np.repeat(w.last_close[:, None], C.H_OUT, 1))


def test_ar_ols_recovers_known_process():
    """AR(4)-OLS on synthetic AR data recovers the coefficients."""
    rng = np.random.default_rng(0)
    phi = np.array([0.3, -0.2, 0.1, 0.05])
    n = 20000
    r = np.zeros(n)
    for t in range(4, n):
        r[t] = phi @ r[t - 4:t][::-1] + rng.normal(scale=0.01)
    close = 10 * np.exp(np.cumsum(r))
    idx = np.arange(0, n - C.WINDOW)[:, None] + np.arange(C.WINDOW)
    w = D.WindowSet(stock=np.zeros(len(idx)), dates=np.zeros(idx.shape), close=close[idx],
                    open=close[idx][:, :5], high=close[idx][:, :5], low=close[idx][:, :5],
                    volume=np.ones((len(idx), 5)))
    sc = D.Scaler(s_ret=0.01, s_vol=1.0, mu_ret=0.0)
    ar = B.AROLS().fit(w, sc)
    np.testing.assert_allclose(ar.coef[1:][::-1], phi, atol=0.03)


def test_diebold_mariano_antisymmetric():
    rng = np.random.default_rng(1)
    a, b = rng.random(500), rng.random(500)
    origin = np.repeat(np.arange(100), 5)
    _, s1, p1 = M.diebold_mariano(a, b, origin)
    _, s2, p2 = M.diebold_mariano(b, a, origin)
    assert np.isclose(s1, -s2) and np.isclose(p1, p2)


def test_closed_loop_first_step_matches_model():
    from stockcast.models import ar_closed_loop, build_vrnn_ar
    import keras
    keras.utils.set_random_seed(0)
    m = build_vrnn_ar(8, 1)
    m.get_layer("head").set_weights([np.ones((8, 1)), np.zeros(1)])
    ctx = np.random.default_rng(0).normal(size=(16, 4, 1)).astype("float32")
    y = ar_closed_loop(m, ctx)
    np.testing.assert_allclose(y[:, 0], np.asarray(m(ctx))[:, -1, 0], rtol=1e-5, atol=1e-6)
    # second step: feed the first prediction back
    ctx2 = np.concatenate([ctx, np.asarray(m(ctx))[:, -1:, :]], 1)
    np.testing.assert_allclose(y[:, 1] - y[:, 0], np.asarray(m(ctx2))[:, -1, 0], rtol=1e-5, atol=1e-6)
