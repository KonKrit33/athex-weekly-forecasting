"""Neural predictors (Keras 3, TensorFlow backend).

VRNN-AR       autoregressive Vanilla RNN (baseline required by the assignment)
                 h_t = tanh(U x_t + W h_{t-1} + b),  r_hat_{t+1} = V h_t + c
              trained with teacher forcing (one-step-ahead), forecasts produced in
              closed loop: each prediction is fed back as the next input.
VRNN-Direct   same Vanilla RNN cell, but the last state is mapped to all 5
              horizons at once (direct / multi-output strategy) - ablation.
Seq2Seq-Attn  bidirectional-GRU encoder + GRU decoder with additive (Bahdanau)
              attention, Lecture 2: s_i = f(s_{i-1}, y_{i-1}, c_i),
              y_i = g(y_{i-1}, s_i, c_i).
Transformer   encoder-decoder Transformer (Lecture 3): sinusoidal positional
              encoding, multi-head self-attention encoder, and a decoder whose
              5 queries are learned "horizon embeddings" that attend to the
              encoded week through encoder-decoder (cross) attention. All 5
              horizons are decoded in parallel (non-autoregressive).

Every network ends with a zero-initialised linear head, i.e. at initialisation
it outputs y_hat = 0, which is exactly the persistence forecast; training can
only move it away from persistence if the data support it.
"""

from __future__ import annotations

import numpy as np
import keras
from keras import layers, ops

from . import config as C


# ==========================================================================
# 1. Vanilla RNNs
# ==========================================================================
def build_vrnn_ar(units: int, layers_n: int = 1, dropout: float = 0.0) -> keras.Model:
    """Sequence-to-sequence one-step-ahead model: (B, T, 1) -> (B, T, 1)."""
    inp = keras.Input((None, 1), name="returns")
    x = inp
    for i in range(layers_n):
        x = layers.SimpleRNN(units, activation="tanh", return_sequences=True,
                             dropout=dropout, name=f"simple_rnn_{i}")(x)
    out = layers.Dense(1, kernel_initializer="zeros", name="head")(x)
    return keras.Model(inp, out, name="vrnn_ar")


def ar_closed_loop(model: keras.Model, context: np.ndarray, horizon: int = C.H_OUT,
                   batch: int = 4096) -> np.ndarray:
    """Recursive (closed-loop) forecasting.

    context: (N, 4, 1) observed returns r_2..r_5. Returns the cumulative
    log-ratio forecasts y_hat_h = sum_{k<=h} r_hat_{5+k}, shape (N, horizon).
    The whole (growing) sequence is re-processed at every step; since the RNN
    is stateless across calls and starts from h_0 = 0, this is identical to
    carrying the hidden state forward.
    """
    out = []
    for i in range(0, len(context), batch):
        seq = context[i:i + batch].astype("float32")
        preds = []
        for _ in range(horizon):
            nxt = model(seq, training=False).numpy()[:, -1:, :]   # r_hat_{t+1}
            preds.append(nxt[:, 0, 0])
            seq = np.concatenate([seq, nxt], axis=1)            # feed back
        out.append(np.cumsum(np.stack(preds, axis=1), axis=1))
    return np.concatenate(out)


def build_vrnn_direct(n_features: int, units: int, layers_n: int = 1,
                      dropout: float = 0.0) -> keras.Model:
    inp = keras.Input((C.L_IN, n_features), name="week")
    x = inp
    for i in range(layers_n):
        x = layers.SimpleRNN(units, activation="tanh", return_sequences=i < layers_n - 1,
                             dropout=dropout, name=f"simple_rnn_{i}")(x)
    out = layers.Dense(C.H_OUT, kernel_initializer="zeros", name="head")(x)
    return keras.Model(inp, out, name="vrnn_direct")


# ==========================================================================
# 2. Encoder-decoder RNN with Bahdanau (additive) attention  (Lecture 2)
# ==========================================================================
@keras.saving.register_keras_serializable(package="stockcast")
class Seq2SeqAttention(keras.Model):
    def __init__(self, units: int = 32, dropout: float = 0.0, horizon: int = C.H_OUT, **kw):
        super().__init__(**kw)
        self.units, self.dropout_rate, self.horizon = units, dropout, horizon
        # encoder: BiRNN -> annotations h_j = [h_fwd_j ; h_bwd_j]
        self.encoder = layers.Bidirectional(
            layers.GRU(units, return_sequences=True, return_state=True), name="bi_gru")
        self.init_state = layers.Dense(units, activation="tanh", name="W_s")   # s_0 = tanh(W_s h_bwd_1)
        # alignment model e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)
        self.W_a = layers.Dense(units, use_bias=False, name="W_a")
        self.U_a = layers.Dense(units, use_bias=False, name="U_a")
        self.v_a = layers.Dense(1, use_bias=False, name="v_a")
        self.cell = layers.GRUCell(units, name="decoder_gru")                  # f
        self.g_hidden = layers.Dense(units, activation="tanh", name="g_hidden")  # g
        self.drop = layers.Dropout(dropout)
        self.head = layers.Dense(1, kernel_initializer="zeros", name="head")

    def call(self, x, training=False, return_attention=False):
        annotations, _h_fwd_T, h_bwd_1 = self.encoder(x, training=training)
        annotations = self.drop(annotations, training=training)
        s = self.init_state(h_bwd_1)
        keys = self.U_a(annotations)                                   # (B, T, u)
        y_prev = ops.zeros_like(s[:, :1])                              # y_0 = z_5 = 0
        ys, alphas = [], []
        for _ in range(self.horizon):
            e = self.v_a(ops.tanh(keys + ops.expand_dims(self.W_a(s), 1)))   # (B, T, 1)
            alpha = ops.softmax(e, axis=1)
            context = ops.sum(alpha * annotations, axis=1)                   # c_i
            s, _ = self.cell(ops.concatenate([y_prev, context], -1), [s], training=training)
            g = self.g_hidden(ops.concatenate([s, context, y_prev], -1))
            y = self.head(self.drop(g, training=training))                   # (B, 1)
            ys.append(y)
            alphas.append(alpha[..., 0])
            y_prev = y        # free-running: the decoder is fed its own prediction
        y = ops.concatenate(ys, axis=1)
        if return_attention:
            return y, ops.stack(alphas, axis=1)                           # (B, H, T)
        return y

    def get_config(self):
        return dict(super().get_config(), units=self.units, dropout=self.dropout_rate,
                    horizon=self.horizon)


def build_seq2seq_attn(n_features: int, units: int, dropout: float = 0.0) -> keras.Model:
    m = Seq2SeqAttention(units=units, dropout=dropout, name="seq2seq_attn")
    m(np.zeros((1, C.L_IN, n_features), "float32"))    # build
    return m


# ==========================================================================
# 3. Transformer encoder-decoder with learned horizon queries  (Lecture 3)
# ==========================================================================
def sinusoidal_encoding(length: int, d_model: int) -> np.ndarray:
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(...)  (Vaswani et al.)."""
    pos = np.arange(length)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return pe.astype("float32")


@keras.saving.register_keras_serializable(package="stockcast")
class EncoderBlock(layers.Layer):
    """Multi-head self-attention + FFN, each wrapped in residual + LayerNorm."""

    def __init__(self, d_model, heads, d_ff, dropout, **kw):
        super().__init__(**kw)
        self.mha = layers.MultiHeadAttention(heads, key_dim=d_model // heads, dropout=dropout)
        self.ffn1 = layers.Dense(d_ff, activation="relu")
        self.ffn2 = layers.Dense(d_model)
        self.ln1, self.ln2 = layers.LayerNormalization(), layers.LayerNormalization()
        self.do1, self.do2 = layers.Dropout(dropout), layers.Dropout(dropout)

    def call(self, x, training=False):
        a, scores = self.mha(x, x, x, return_attention_scores=True, training=training)
        x = self.ln1(x + self.do1(a, training=training))
        f = self.ffn2(self.ffn1(x))
        return self.ln2(x + self.do2(f, training=training)), scores


@keras.saving.register_keras_serializable(package="stockcast")
class DecoderBlock(layers.Layer):
    """Self-attention among horizon queries, cross-attention to the encoder, FFN.

    No causal mask: the 5 horizons are produced in parallel (non-autoregressive),
    so every query may look at every other query.
    """

    def __init__(self, d_model, heads, d_ff, dropout, **kw):
        super().__init__(**kw)
        self.self_mha = layers.MultiHeadAttention(heads, key_dim=d_model // heads, dropout=dropout)
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim=d_model // heads, dropout=dropout)
        self.ffn1 = layers.Dense(d_ff, activation="relu")
        self.ffn2 = layers.Dense(d_model)
        self.ln1, self.ln2, self.ln3 = (layers.LayerNormalization() for _ in range(3))
        self.do1, self.do2, self.do3 = (layers.Dropout(dropout) for _ in range(3))

    def call(self, q, memory, training=False):
        a = self.self_mha(q, q, q, training=training)
        q = self.ln1(q + self.do1(a, training=training))
        c, scores = self.cross_mha(q, memory, memory, return_attention_scores=True,
                                   training=training)
        q = self.ln2(q + self.do2(c, training=training))
        f = self.ffn2(self.ffn1(q))
        return self.ln3(q + self.do3(f, training=training)), scores


@keras.saving.register_keras_serializable(package="stockcast")
class HorizonTransformer(keras.Model):
    def __init__(self, n_features, d_model=32, heads=4, layers_n=1, ff_mult=2,
                 dropout=0.1, horizon=C.H_OUT, **kw):
        super().__init__(**kw)
        self.cfg = dict(n_features=n_features, d_model=d_model, heads=heads,
                        layers_n=layers_n, ff_mult=ff_mult, dropout=dropout, horizon=horizon)
        self.embed = layers.Dense(d_model, name="input_embedding")
        self.pe = ops.convert_to_tensor(sinusoidal_encoding(C.L_IN, d_model))
        self.drop = layers.Dropout(dropout)
        self.enc = [EncoderBlock(d_model, heads, ff_mult * d_model, dropout, name=f"enc_{i}")
                    for i in range(layers_n)]
        self.dec = [DecoderBlock(d_model, heads, ff_mult * d_model, dropout, name=f"dec_{i}")
                    for i in range(layers_n)]
        self.horizon = horizon
        self.head = layers.Dense(1, kernel_initializer="zeros", name="head")

    def build(self, input_shape):
        d = self.cfg["d_model"]
        self.queries = self.add_weight(shape=(self.horizon, d), name="horizon_queries",
                                       initializer=keras.initializers.RandomNormal(stddev=0.5))
        super().build(input_shape)

    def call(self, x, training=False, return_attention=False):
        h = self.embed(x) * np.sqrt(self.cfg["d_model"]) + self.pe
        h = self.drop(h, training=training)
        enc_scores = []
        for blk in self.enc:
            h, s = blk(h, training=training)
            enc_scores.append(s)
        q = ops.broadcast_to(self.queries[None], (ops.shape(x)[0], self.horizon, self.cfg["d_model"]))
        cross_scores = []
        for blk in self.dec:
            q, s = blk(q, h, training=training)
            cross_scores.append(s)
        y = ops.squeeze(self.head(q), -1)                               # (B, H)
        if return_attention:
            return y, enc_scores, cross_scores
        return y

    def get_config(self):
        return dict(super().get_config(), **self.cfg)


def build_transformer(n_features: int, d_model: int, heads: int, layers: int,
                      ff_mult: int = 2, dropout: float = 0.1) -> keras.Model:
    m = HorizonTransformer(n_features, d_model, heads, layers, ff_mult, dropout,
                           name="transformer")
    m(np.zeros((1, C.L_IN, n_features), "float32"))
    return m


# ==========================================================================
# Registry
# ==========================================================================
def build(family: str, n_features: int, hp: dict) -> keras.Model:
    if family == "vrnn_ar":
        return build_vrnn_ar(hp["units"], hp["layers"], hp.get("dropout", 0.0))
    if family == "vrnn_direct":
        return build_vrnn_direct(n_features, hp["units"], hp["layers"], hp.get("dropout", 0.0))
    if family == "seq2seq_attn":
        return build_seq2seq_attn(n_features, hp["units"], hp.get("dropout", 0.0))
    if family == "transformer":
        return build_transformer(n_features, hp["d_model"], hp["heads"], hp["layers"],
                                 hp.get("ff_mult", 2), hp.get("dropout", 0.1))
    raise ValueError(family)
