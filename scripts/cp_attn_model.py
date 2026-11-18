"""
CP-Attn-Lite: dual-branch 1D-CNN with cyber–physical fusion attention.

Does not replace the baseline Sequential 1D-CNN. Cyber (21) and physical (11)
features are encoded separately, fused with a learned 2-way gate, then
channel-gated and classified into 5 attack classes.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import Model, layers

CYBER_DIM = 21
PHYS_DIM = 11
FILTERS = 32


@tf.keras.utils.register_keras_serializable(package="uav_ids")
class FusionGate(layers.Layer):
    """Softmax gate: how much of this sample is cyber vs physical."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(2, name="fusion_score")

    def call(self, inputs):
        cyber, physical = inputs
        weights = tf.nn.softmax(self.score(tf.concat([cyber, physical], axis=-1)), axis=-1)
        fused = tf.concat(
            [cyber * weights[:, 0:1], physical * weights[:, 1:2]],
            axis=-1,
        )
        return fused, weights

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="uav_ids")
class SqueezeExcite(layers.Layer):
    """Channel gate on the fused vector (which learned filters matter)."""

    def __init__(self, units: int, ratio: int = 4, **kwargs):
        super().__init__(**kwargs)
        hidden = max(units // ratio, 4)
        self.units = units
        self.ratio = ratio
        self.squeeze = layers.Dense(hidden, activation="relu")
        self.excite = layers.Dense(units, activation="sigmoid")

    def call(self, x):
        return x * self.excite(self.squeeze(x))

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "ratio": self.ratio})
        return config


def _branch(x, name: str):
    x = layers.Conv1D(FILTERS, 3, padding="same", activation="relu", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)


def build_cp_attn_model() -> Model:
    cyber_in = layers.Input(shape=(CYBER_DIM, 1), name="cyber")
    phys_in = layers.Input(shape=(PHYS_DIM, 1), name="physical")
    cyber_vec = _branch(cyber_in, "cyber")
    phys_vec = _branch(phys_in, "physical")
    fused, weights = FusionGate(name="fusion_gate")([cyber_vec, phys_vec])
    gated = SqueezeExcite(FILTERS * 2, name="channel_gate")(fused)
    gated = layers.Dropout(0.3)(gated)
    probs = layers.Dense(5, activation="softmax", name="probs")(gated)
    return Model([cyber_in, phys_in], [probs, weights], name="cp_attn_lite")


def split_modalities(features: np.ndarray, fusion: str) -> tuple[np.ndarray, np.ndarray]:
    """Map a fusion feature matrix into (cyber, physical) tensors shaped (N, L, 1)."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = x.shape[0]
    kind = "combined"
    if fusion.endswith("_cyber"):
        kind = "cyber"
    elif fusion.endswith("_physical"):
        kind = "physical"

    if kind == "cyber":
        if x.shape[1] != CYBER_DIM:
            raise ValueError(f"Expected {CYBER_DIM} cyber features, got {x.shape[1]}")
        cyber = x.reshape(n, CYBER_DIM, 1)
        physical = np.zeros((n, PHYS_DIM, 1), dtype=np.float32)
    elif kind == "physical":
        if x.shape[1] != PHYS_DIM:
            raise ValueError(f"Expected {PHYS_DIM} physical features, got {x.shape[1]}")
        physical = x.reshape(n, PHYS_DIM, 1)
        cyber = np.zeros((n, CYBER_DIM, 1), dtype=np.float32)
    else:
        if x.shape[1] < CYBER_DIM + PHYS_DIM:
            raise ValueError(f"Expected {CYBER_DIM + PHYS_DIM} combined features, got {x.shape[1]}")
        cyber = x[:, :CYBER_DIM].reshape(n, CYBER_DIM, 1)
        physical = x[:, CYBER_DIM:CYBER_DIM + PHYS_DIM].reshape(n, PHYS_DIM, 1)
    return cyber, physical
