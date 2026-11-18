"""
Uncertainty Estimation Module for UAV Cyber-Physical Intrusion Detection.

Implements:
  1. Normalized Predictive Entropy (Shannon entropy capturing total uncertainty).
  2. Monte Carlo Dropout (MC Dropout) variance / mutual information to isolate
     epistemic (model) uncertainty from aleatoric (data) uncertainty.
  3. Branch-level predictive uncertainty calculations for cyber and physical branches.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def compute_normalized_entropy(probs: np.ndarray | tf.Tensor, num_classes: int = 5) -> np.ndarray:
    """
    Computes Shannon entropy normalized to [0, 1] range:
      H_norm(p) = - sum(p * ln(p)) / ln(K)
    0.0 = completely certain (one-hot probability).
    1.0 = maximum uncertainty (uniform distribution).
    """
    if isinstance(probs, tf.Tensor):
        p = tf.clip_by_value(probs, 1e-12, 1.0)
        entropy = -tf.reduce_sum(p * tf.math.log(p), axis=-1)
        max_entropy = tf.math.log(float(num_classes))
        return (entropy / max_entropy).numpy()

    p = np.clip(np.asarray(probs, dtype=np.float32), 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)
    max_entropy = np.log(float(num_classes))
    return np.clip(entropy / max_entropy, 0.0, 1.0)


def compute_mc_dropout_uncertainty(
    model: tf.keras.Model,
    inputs: list[np.ndarray],
    n_passes: int = 10,
    num_classes: int = 5,
) -> dict[str, np.ndarray]:
    """
    Performs Monte Carlo Dropout inference by executing multiple stochastic forward passes
    with dropout activated at test time.

    Decomposes uncertainty into:
      1. Total Predictive Uncertainty (Shannon entropy of mean prediction).
      2. Epistemic Uncertainty (Mutual Information = Total Entropy - Expected Entropy).
      3. Aleatoric Uncertainty (Expected Entropy across stochastic passes).
      4. Predictive Variance (Mean variance across classes).
    """
    preds_list = []
    gate_list = []

    for _ in range(n_passes):
        # Pass training=True to keep Dropout layers active
        out = model(inputs, training=True)
        if isinstance(out, (list, tuple)):
            probs = out[0]
            if len(out) > 1:
                gate_list.append(out[1].numpy())
        else:
            probs = out
        preds_list.append(probs.numpy())

    # Stack: shape (T, N, K)
    stacked_preds = np.stack(preds_list, axis=0)

    # 1. Mean prediction across passes: (N, K)
    mean_probs = np.mean(stacked_preds, axis=0)

    # 2. Total predictive uncertainty: H(E[p])
    total_entropy = compute_normalized_entropy(mean_probs, num_classes=num_classes)

    # 3. Expected entropy: E[H(p)] (Aleatoric uncertainty proxy)
    entropies_per_pass = np.array([compute_normalized_entropy(p, num_classes=num_classes) for p in stacked_preds])
    expected_entropy = np.mean(entropies_per_pass, axis=0)

    # 4. Epistemic Uncertainty (Mutual Information = Total Entropy - Expected Entropy)
    epistemic_mi = np.clip(total_entropy - expected_entropy, 0.0, 1.0)

    # 5. Class variance across passes: (N, K) -> (N,)
    var_per_class = np.var(stacked_preds, axis=0)
    mean_variance = np.mean(var_per_class, axis=-1)

    # Mean gate weights if available
    mean_gates = np.mean(np.stack(gate_list, axis=0), axis=0) if gate_list else None

    return {
        "mean_probs": mean_probs,
        "total_uncertainty": total_entropy,
        "epistemic_uncertainty": epistemic_mi,
        "aleatoric_uncertainty": expected_entropy,
        "predictive_variance": mean_variance,
        "mean_gate_weights": mean_gates,
    }
