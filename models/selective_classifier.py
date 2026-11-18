"""
Selective Prediction & Temperature Calibration Engine for RA-SAFE.

Implements:
  1. Post-hoc Temperature Scaling to calibrate predictive softmax probabilities.
  2. Validation-based threshold determination for confidence (tau_conf) and reliability (tau_rel).
  3. Selective Prediction Decision Rule (Accept vs Abstain).
  4. Operator Action Recommendation Playbook for abstained/degraded flight states.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


class SelectiveClassifier:
    """Manages probability calibration, threshold selection, and selective prediction."""

    def __init__(
        self,
        temperature: float = 1.0,
        conf_threshold: float = 0.70,
        rel_threshold: float = 0.30,
    ) -> None:
        self.temperature: float = temperature
        self.conf_threshold: float = conf_threshold
        self.rel_threshold: float = rel_threshold
        self.is_calibrated: bool = False

    def fit_temperature(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Fit temperature T > 0 on validation set logits using Negative Log-Likelihood (NLL).
        Guarantees NO test data is used for calibration.
        """
        val_logits = np.asarray(val_logits, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=int)

        def nll_objective(t: np.ndarray) -> float:
            scaled_logits = val_logits / max(t[0], 1e-4)
            # numerically stable softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return log_loss(val_labels, probs, labels=list(range(5)))

        res = minimize(nll_objective, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        self.is_calibrated = True
        return self.temperature

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling to raw logits."""
        scaled = np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-4)
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
        return probs.astype(np.float32)

    def tune_thresholds_on_validation(
        self,
        val_probs: np.ndarray,
        val_reliabilities: np.ndarray,
        val_labels: np.ndarray,
        target_max_risk: float = 0.05,
        min_acceptable_coverage: float = 0.80,
    ) -> tuple[float, float]:
        """
        Determine optimal confidence and reliability thresholds on validation data.
        Finds highest coverage such that selective risk (error on accepted) <= target_max_risk.
        """
        confs = np.max(val_probs, axis=-1)
        preds = np.argmax(val_probs, axis=-1)
        max_rel = np.max(val_reliabilities, axis=-1)

        best_conf = 0.65
        best_rel = 0.25
        best_coverage = 0.0

        conf_grid = np.linspace(0.40, 0.95, 23)
        rel_grid = np.linspace(0.10, 0.50, 9)

        for c_th in conf_grid:
            for r_th in rel_grid:
                accepted_mask = (confs >= c_th) & (max_rel >= r_th)
                n_acc = np.sum(accepted_mask)
                if n_acc < 10:
                    continue
                cov = n_acc / len(val_labels)
                risk = np.mean(preds[accepted_mask] != val_labels[accepted_mask])

                if risk <= target_max_risk and cov >= min_acceptable_coverage:
                    if cov > best_coverage:
                        best_coverage = cov
                        best_conf = float(c_th)
                        best_rel = float(r_th)

        self.conf_threshold = best_conf
        self.rel_threshold = best_rel
        return best_conf, best_rel

    def predict_sample(
        self,
        logits: np.ndarray,
        r_c: float,
        r_p: float,
        w_c: float,
        w_p: float,
        class_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single sample through the calibrated selective prediction engine.
        Returns detailed decision, probabilities, uncertainty, and operator action.
        """
        if class_names is None:
            class_names = ["benign", "dos", "replay", "evil_twin", "fdi"]

        cal_probs = self.calibrate_probs(logits.reshape(1, -1))[0]
        pred_idx = int(np.argmax(cal_probs))
        confidence = float(cal_probs[pred_idx])

        # Normalized Shannon entropy as calibrated uncertainty
        eps = 1e-12
        entropy = -np.sum(cal_probs * np.log(cal_probs + eps)) / np.log(len(class_names))
        uncertainty = float(entropy)

        max_rel = max(float(r_c), float(r_p))
        accepted = (confidence >= self.conf_threshold) and (max_rel >= self.rel_threshold)

        if accepted:
            decision = "ACCEPTED"
            predicted_label = class_names[pred_idx]
            operator_recommendation = (
                f"Automated detection confirmed ({predicted_label.upper()}). "
                "Confidence and branch reliability meet operational safety criteria."
            )
        else:
            decision = "ABSTAINED"
            predicted_label = "Uncertain — Human Review Required"
            # Diagnosing specific failure mode for operator playbook
            if r_c < self.rel_threshold and r_p >= self.rel_threshold:
                operator_recommendation = (
                    "CAUTION: Wi-Fi Cyber link severely degraded or spoofed (R_c low). "
                    "Inertial physical telemetry is reliable. Disregard network telemetry, "
                    "verify C2 link security, and rely on autonomous onboard navigation."
                )
            elif r_p < self.rel_threshold and r_c >= self.rel_threshold:
                operator_recommendation = (
                    "CAUTION: Physical telemetry anomalies or sensor freeze detected (R_p low). "
                    "Network channel is clear. Cross-check IMU/compass telemetry and initiate "
                    "stabilized hover or altitude hold."
                )
            elif r_c < self.rel_threshold and r_p < self.rel_threshold:
                operator_recommendation = (
                    "CRITICAL: Dual-domain failure. Cyber link and physical sensor health both severely "
                    "compromised. Immediately engage hardware Return-to-Home (RTH) failsafe or safe parachute descent."
                )
            else:
                operator_recommendation = (
                    f"UNCERTAIN: Attack prediction ambiguous (Confidence {confidence:.1%} < Threshold {self.conf_threshold:.1%}). "
                    "Operator manual verification required before executing countermeasures."
                )

        return {
            "decision": decision,
            "predicted_class_id": pred_idx if accepted else -1,
            "predicted_class_name": predicted_label,
            "raw_predicted_class_id": pred_idx,
            "raw_predicted_class_name": class_names[pred_idx],
            "confidence": confidence,
            "uncertainty": uncertainty,
            "calibrated_probabilities": cal_probs.tolist(),
            "cyber_reliability": float(r_c),
            "phys_reliability": float(r_p),
            "cyber_weight": float(w_c),
            "phys_weight": float(w_p),
            "operator_recommendation": operator_recommendation,
        }

    def evaluate_selective_performance(
        self,
        test_probs: np.ndarray,
        test_reliabilities: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute coverage, selective risk, accuracy on accepted, and abstention rate."""
        confs = np.max(test_probs, axis=-1)
        preds = np.argmax(test_probs, axis=-1)
        max_rel = np.max(test_reliabilities, axis=-1)

        accepted_mask = (confs >= self.conf_threshold) & (max_rel >= self.rel_threshold)
        n_total = len(test_labels)
        n_acc = int(np.sum(accepted_mask))
        n_abst = n_total - n_acc

        coverage = float(n_acc / n_total) if n_total > 0 else 0.0
        abstention_rate = float(n_abst / n_total) if n_total > 0 else 0.0

        if n_acc > 0:
            acc_correct = np.sum(preds[accepted_mask] == test_labels[accepted_mask])
            acc_accuracy = float(acc_correct / n_acc)
            selective_risk = float(1.0 - acc_accuracy)
        else:
            acc_accuracy = 0.0
            selective_risk = 0.0

        overall_acc = float(np.mean(preds == test_labels))

        return {
            "coverage": coverage,
            "abstention_rate": abstention_rate,
            "accepted_samples": n_acc,
            "abstained_samples": n_abst,
            "accepted_accuracy": acc_accuracy,
            "selective_risk": selective_risk,
            "overall_accuracy_unfiltered": overall_acc,
            "conf_threshold": self.conf_threshold,
            "rel_threshold": self.rel_threshold,
            "temperature": self.temperature,
        }

    def save(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature,
            "conf_threshold": self.conf_threshold,
            "rel_threshold": self.rel_threshold,
            "is_calibrated": self.is_calibrated,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "SelectiveClassifier":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = cls(
            temperature=float(data.get("temperature", 1.0)),
            conf_threshold=float(data.get("conf_threshold", 0.70)),
            rel_threshold=float(data.get("rel_threshold", 0.30)),
        )
        inst.is_calibrated = data.get("is_calibrated", True)
        return inst
