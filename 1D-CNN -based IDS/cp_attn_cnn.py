"""
Train CP-Attn-Lite (dual-branch fusion-attention 1D-CNN).

Baseline multiclass_cnn.py is unchanged.

Run from project root:
  python "1D-CNN -based IDS/cp_attn_cnn.py" --fusion central_combined --epochs 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.attack_labels import ATTACK_CLASS_NAMES, ATTACK_DISPLAY_NAMES, NUM_ATTACK_CLASSES  # noqa: E402
from scripts.cp_attn_model import build_cp_attn_model, split_modalities  # noqa: E402
from scripts.multiclass_data import frame_to_xy, load_fusion_frame  # noqa: E402
from scripts.save_multiclass_metrics import save_multiclass_metrics  # noqa: E402

CLASS_ORDER = [ATTACK_CLASS_NAMES[i] for i in range(NUM_ATTACK_CLASSES)]
DISPLAY_ORDER = [ATTACK_DISPLAY_NAMES[i] for i in range(NUM_ATTACK_CLASSES)]
MODEL_DIR = Path(__file__).resolve().parent / "cp_attn_models"
RESULTS_DIR = Path(__file__).resolve().parent / "cp_attn_results"


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ticks = np.arange(len(DISPLAY_ORDER))
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=DISPLAY_ORDER,
        yticklabels=DISPLAY_ORDER,
        ylabel="True class",
        xlabel="Predicted class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fusion_by_class(y_true: np.ndarray, weights: np.ndarray) -> dict[str, dict[str, float]]:
    """Mean cyber/physical gate weight for each true class."""
    out: dict[str, dict[str, float]] = {}
    for class_id, name in enumerate(CLASS_ORDER):
        mask = y_true == class_id
        if not np.any(mask):
            continue
        out[name] = {
            "cyber_weight": float(weights[mask, 0].mean()),
            "physical_weight": float(weights[mask, 1].mean()),
            "support": int(mask.sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CP-Attn-Lite multi-class IDS")
    parser.add_argument("--fusion", default="central_combined")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--metrics_title", default=None)
    args = parser.parse_args()

    fusion = args.fusion
    metrics_title = args.metrics_title or fusion
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_fusion_frame(fusion)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)

    x, y = frame_to_xy(df)
    counts = np.bincount(y, minlength=NUM_ATTACK_CLASSES)
    split_kw = {"test_size": 0.2, "random_state": 42}
    if counts[counts > 0].min() >= 2:
        split_kw["stratify"] = y
    x_train, x_test, y_train, y_test = train_test_split(x, y, **split_kw)

    cyber_train, phys_train = split_modalities(x_train, fusion)
    cyber_test, phys_test = split_modalities(x_test, fusion)
    y_train_oh = to_categorical(y_train, num_classes=NUM_ATTACK_CLASSES)
    dummy_train = np.zeros((len(y_train), 2), dtype=np.float32)

    model = build_cp_attn_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=["categorical_crossentropy", "mse"],
        loss_weights=[1.0, 0.0],
    )
    print(f"CP-Attn-Lite parameters: {model.count_params():,}")
    model.fit(
        [cyber_train, phys_train],
        [y_train_oh, dummy_train],
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=64,
        verbose=1,
    )

    out_path = MODEL_DIR / f"cp_attn_{fusion}.keras"
    model.save(out_path)
    print(f"Saved model: {out_path}")

    t0 = time.perf_counter()
    probs, weights = model.predict([cyber_test, phys_test], verbose=0, batch_size=256)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_sample_ms = elapsed_ms / max(len(y_test), 1)
    y_pred = np.argmax(probs, axis=-1)

    label_ids = list(range(NUM_ATTACK_CLASSES))
    print(classification_report(y_test, y_pred, labels=label_ids, target_names=CLASS_ORDER, zero_division=0))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_test, y_pred, labels=label_ids)
    plot_confusion(cm, RESULTS_DIR / f"confusion_{fusion}.png", f"CP-Attn-Lite — {metrics_title}")
    gates = fusion_by_class(y_test, weights)

    report = classification_report(
        y_test, y_pred, labels=label_ids, target_names=CLASS_ORDER, output_dict=True, zero_division=0,
    )
    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in CLASS_ORDER
        if name in report
    }

    save_multiclass_metrics(
        model="CP-Attn-Lite",
        fusion=metrics_title,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        auc_macro=auc,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        class_names=CLASS_ORDER,
        extras={
            "n_params": int(model.count_params()),
            "inference_ms_per_sample": float(per_sample_ms),
            "fusion_by_class": gates,
        },
    )
    print(f"Macro F1: {f1_macro:.4f}  Accuracy: {accuracy:.4f}  {per_sample_ms:.2f} ms/sample")
    for name, row in gates.items():
        print(f"  {name:12s}  cyber={row['cyber_weight']:.2f}  physical={row['physical_weight']:.2f}")


if __name__ == "__main__":
    main()
