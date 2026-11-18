"""
Multi-class cyber-physical IDS (5 attack types) using 1D-CNN.

Does not modify binary 1D_CNN.py — separate entry point and metrics file.

Run from project root:
  python "1D-CNN -based IDS/multiclass_cnn.py" --fusion drone1_combined
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv1D, Dense, Dropout, Flatten
from keras.models import Sequential
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
from scripts.save_multiclass_metrics import save_multiclass_metrics  # noqa: E402

CLASS_ORDER = [ATTACK_CLASS_NAMES[i] for i in range(NUM_ATTACK_CLASSES)]
DISPLAY_ORDER = [ATTACK_DISPLAY_NAMES[i] for i in range(NUM_ATTACK_CLASSES)]

parser = argparse.ArgumentParser(description="Multi-class 1D-CNN IDS")
parser.add_argument("--fusion", default="drone1_combined", help="Fusion dataset key")
parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
parser.add_argument("--max_samples", type=int, default=0, help="Subsample rows (0 = all)")
parser.add_argument("--metrics_title", default=None, help="Label for saved metrics")
parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

fusion = args.fusion
metrics_title = args.metrics_title or fusion
os.chdir(Path(__file__).resolve().parent)
os.makedirs("multiclass_results", exist_ok=True)
os.makedirs("multiclass_models", exist_ok=True)

from scripts.multiclass_data import frame_to_xy, load_fusion_frame, model_path  # noqa: E402

df = load_fusion_frame(fusion)
if args.max_samples and len(df) > args.max_samples:
    df = df.sample(n=args.max_samples, random_state=42)

X, y = frame_to_xy(df)

class_counts = np.bincount(y, minlength=NUM_ATTACK_CLASSES)
use_stratify = class_counts[class_counts > 0].min() >= 2
split_kw = {"test_size": 0.2, "random_state": 42}
if use_stratify:
    split_kw["stratify"] = y
X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kw)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_train_one_hot = to_categorical(y_train, num_classes=NUM_ATTACK_CLASSES)
y_test_one_hot = to_categorical(y_test, num_classes=NUM_ATTACK_CLASSES)


def create_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(Conv1D(64, 3, activation="relu", input_shape=input_shape))
    model.add(Conv1D(128, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(NUM_ATTACK_CLASSES, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(DISPLAY_ORDER))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=DISPLAY_ORDER,
        yticklabels=DISPLAY_ORDER,
        ylabel="True class",
        xlabel="Predicted class",
        title=f"Multi-class confusion matrix — {metrics_title}",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(model: Sequential) -> None:
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=-1)

    label_ids = list(range(NUM_ATTACK_CLASSES))
    present = sorted(set(y_test) | set(y_pred))
    print(f"Classes in this fusion split: {[CLASS_ORDER[i] for i in present]}")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=label_ids,
            target_names=CLASS_ORDER,
            zero_division=0,
        )
    )

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_ATTACK_CLASSES)))
    cm_path = Path("multiclass_results") / f"confusion_{fusion}.png"
    plot_confusion(cm, cm_path)

    report = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
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
        model="1D-CNN (multi-class)",
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
    )
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Macro F1: {f1_macro:.4f}  Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    model = create_model((X_train.shape[1], 1))
    model.fit(
        X_train,
        y_train_one_hot,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=64,
        verbose=1,
    )
    if args.save_model:
        out = model_path(fusion)
        model.save(out)
        print(f"Saved streaming model: {out}")
    evaluate(model)
