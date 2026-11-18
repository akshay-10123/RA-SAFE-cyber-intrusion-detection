"""
Calibration and Uncertainty Evaluation Metrics for UAV Intrusion Detection.

Implements:
  1. Expected Calibration Error (ECE) with equal-width confidence binning.
  2. Multi-class Brier Score.
  3. Negative Log-Likelihood (NLL).
  4. Empirical Risk-Coverage Curve calculation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def expected_calibration_error(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE):
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=np.float32)

    confidences = np.max(y_probs, axis=-1)
    predictions = np.argmax(y_probs, axis=-1)
    accuracies = (predictions == y_true).astype(float)

    n_samples = len(y_true)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 5) -> float:
    """
    Multi-class Brier score:
      BS = (1 / N) * sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2
    Lower is better (0.0 = perfect probabilistic calibration).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=np.float32)
    n = len(y_true)

    # One-hot encode targets
    y_one_hot = np.zeros((n, num_classes), dtype=np.float32)
    y_one_hot[np.arange(n), y_true] = 1.0

    return float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))


def negative_log_likelihood(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Negative Log-Likelihood (Categorical Cross-Entropy) over all samples."""
    eps = 1e-12
    p = np.clip(np.asarray(y_probs, dtype=np.float64), eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return float(log_loss(y_true, p, labels=list(range(p.shape[1]))))


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    reliabilities: np.ndarray | None = None,
    n_points: int = 30,
) -> dict[str, list[float]]:
    """
    Generates a Risk-Coverage curve by varying the rejection threshold from lenient to strict.
    Coverage: fraction of accepted samples.
    Risk: classification error on accepted samples.
    """
    y_true = np.asarray(y_true, dtype=int)
    confs = np.max(y_probs, axis=-1)
    preds = np.argmax(y_probs, axis=-1)

    thresholds = np.linspace(0.20, 0.98, n_points)
    coverages = []
    risks = []

    for th in thresholds:
        if reliabilities is not None:
            max_rel = np.max(reliabilities, axis=-1)
            accepted = (confs >= th) & (max_rel >= 0.20)
        else:
            accepted = confs >= th

        n_acc = np.sum(accepted)
        if n_acc == 0:
            coverages.append(0.0)
            risks.append(0.0)
            continue

        cov = float(n_acc / len(y_true))
        risk = float(np.mean(preds[accepted] != y_true[accepted]))
        coverages.append(cov)
        risks.append(risk)

    return {
        "thresholds": [float(t) for t in thresholds],
        "coverages": coverages,
        "risks": risks,
    }
