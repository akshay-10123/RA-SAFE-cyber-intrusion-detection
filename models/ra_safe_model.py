"""
RA-SAFE: Reliability-Aware Selective Fusion for UAV Cyber-Physical Intrusion Detection.

Extends CP-Attn-Lite with:
  1. Input-quality indicators Q_c and Q_p.
  2. Auxiliary classification heads for branch predictive uncertainty U_c and U_p.
  3. Integrated branch reliability scores R_c and R_p.
  4. Learnable 70-dimensional reliability-aware adaptive fusion gate.
  5. Squeeze-and-Excitation channel gating.
  6. Final calibrated classification head with selective prediction support.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import Model, layers

CYBER_DIM = 21
PHYS_DIM = 11
FILTERS = 32
NUM_CLASSES = 5


@tf.keras.utils.register_keras_serializable(package="ra_safe")
class ReliabilityAdaptiveFusionGate(layers.Layer):
    """
    Learnable reliability-aware gating mechanism.
    Fuses latent feature representations modulated by data quality (Q) and uncertainty (U).
    """

    def __init__(self, hidden_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.fc1 = layers.Dense(hidden_dim, activation="relu", name="gate_fc1")
        self.fc2 = layers.Dense(2, name="gate_fc2")

    def call(self, inputs):
        # inputs: [z_c, z_p, q_c, q_p, c_probs, p_probs]
        z_c, z_p, q_c, q_p, c_probs, p_probs = inputs
        eps = 1e-10
        norm_factor = tf.math.log(float(NUM_CLASSES))

        # 1. Branch predictive uncertainties (normalized Shannon entropy)
        u_c = -tf.reduce_sum(c_probs * tf.math.log(c_probs + eps), axis=-1, keepdims=True) / norm_factor
        u_p = -tf.reduce_sum(p_probs * tf.math.log(p_probs + eps), axis=-1, keepdims=True) / norm_factor
        u_c = tf.clip_by_value(u_c, 0.0, 1.0)
        u_p = tf.clip_by_value(u_p, 0.0, 1.0)

        # 2. Unified branch reliability scores R_c = Q_c * (1 - U_c), R_p = Q_p * (1 - U_p)
        r_c = q_c * (1.0 - u_c)
        r_p = q_p * (1.0 - u_p)

        # 3. Form 70-dim gating vector: [z_c (32), z_p (32), Q_c (1), Q_p (1), U_c (1), U_p (1), R_c (1), R_p (1)]
        gate_input = tf.concat([z_c, z_p, q_c, q_p, u_c, u_p, r_c, r_p], axis=-1)

        # 4. Compute adaptive fusion weights [w_c, w_p]
        scores = self.fc2(self.fc1(gate_input))
        weights = tf.nn.softmax(scores, axis=-1)  # shape (B, 2)

        # 5. Weighted feature fusion
        w_c = weights[:, 0:1]
        w_p = weights[:, 1:2]
        fused = tf.concat([z_c * w_c, z_p * w_p], axis=-1)  # shape (B, 64)

        r_pair = tf.concat([r_c, r_p], axis=-1)  # shape (B, 2)
        u_pair = tf.concat([u_c, u_p], axis=-1)  # shape (B, 2)

        return fused, weights, r_pair, u_pair

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="ra_safe")
class SqueezeExcite(layers.Layer):
    """Channel attention gate on fused cyber-physical vector."""

    def __init__(self, units: int = 64, ratio: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.ratio = ratio
        hidden = max(units // ratio, 4)
        self.squeeze = layers.Dense(hidden, activation="relu", name="se_squeeze")
        self.excite = layers.Dense(units, activation="sigmoid", name="se_excite")

    def call(self, x):
        return x * self.excite(self.squeeze(x))

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "ratio": self.ratio})
        return config


def _conv_branch(x, name: str):
    """1D-CNN feature extractor for one modality branch."""
    x = layers.Conv1D(FILTERS, 3, padding="same", activation="relu", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)


def build_ra_safe_model() -> Model:
    """
    Constructs the end-to-end RA-SAFE computational graph.
    Inputs:
      - cyber: (21, 1)
      - physical: (11, 1)
      - q_cyber: (1,)
      - q_physical: (1,)
    Outputs:
      - probs: (5,)
      - weights: (2,)  [w_cyber, w_physical]
      - reliabilities: (2,)  [R_cyber, R_physical]
      - uncertainties: (2,)  [U_cyber, U_physical]
      - cyber_aux: (5,)
      - phys_aux: (5,)
      - logits: (5,)
    """
    cyber_in = layers.Input(shape=(CYBER_DIM, 1), name="cyber")
    phys_in = layers.Input(shape=(PHYS_DIM, 1), name="physical")
    q_cyber_in = layers.Input(shape=(1,), name="q_cyber")
    q_phys_in = layers.Input(shape=(1,), name="q_physical")

    # 1. Feature Encoders
    z_c = _conv_branch(cyber_in, "cyber")  # (B, 32)
    z_p = _conv_branch(phys_in, "physical")  # (B, 32)

    # 2. Auxiliary Classification Heads for Modality-Specific Uncertainty
    cyber_aux = layers.Dense(NUM_CLASSES, activation="softmax", name="cyber_aux")(z_c)
    phys_aux = layers.Dense(NUM_CLASSES, activation="softmax", name="phys_aux")(z_p)

    # 3. Reliability-Aware Adaptive Fusion Gate
    gate_layer = ReliabilityAdaptiveFusionGate(hidden_dim=16, name="fusion_gate")
    fused, weights, r_pair, u_pair = gate_layer([z_c, z_p, q_cyber_in, q_phys_in, cyber_aux, phys_aux])

    weights = layers.Identity(name="weights")(weights)
    r_pair = layers.Identity(name="r_pair")(r_pair)
    u_pair = layers.Identity(name="u_pair")(u_pair)

    # 4. Channel Squeeze-and-Excitation
    gated = SqueezeExcite(FILTERS * 2, ratio=4, name="channel_gate")(fused)
    dropped = layers.Dropout(0.3, name="drop_head")(gated)

    # 5. Final Multi-Class Classifier
    logits = layers.Dense(NUM_CLASSES, name="logits")(dropped)
    probs = layers.Softmax(name="probs")(logits)

    model = Model(
        inputs=[cyber_in, phys_in, q_cyber_in, q_phys_in],
        outputs=[probs, weights, r_pair, u_pair, cyber_aux, phys_aux, logits],
        name="ra_safe",
    )
    return model
