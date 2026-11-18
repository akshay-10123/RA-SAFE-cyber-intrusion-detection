"""
Simulated live IDS stream — row-by-row inference from a saved multi-class model.

Example:
  python scripts/live_stream_demo.py --fusion central_combined --start 13700 --steps 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.attack_info import get_attack_info, is_intrusion  # noqa: E402
from scripts.attack_labels import ATTACK_CLASS_NAMES, ATTACK_DISPLAY_NAMES  # noqa: E402
from scripts.multiclass_data import (  # noqa: E402
    frame_to_xy,
    load_fusion_frame,
    model_path,
    reshape_for_cnn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live stream simulation (CLI)")
    parser.add_argument("--fusion", default="central_combined")
    parser.add_argument("--start", type=int, default=0, help="Start row index in dataset")
    parser.add_argument("--steps", type=int, default=100, help="Number of rows to stream")
    parser.add_argument("--delay-ms", type=float, default=50.0, help="Delay between rows")
    parser.add_argument("--model", default=None, help="Override model .keras path")
    args = parser.parse_args()

    import tensorflow as tf

    mpath = Path(args.model) if args.model else model_path(args.fusion)
    if not mpath.exists():
        print(f"Model not found: {mpath}")
        print(f'Train first: python "1D-CNN -based IDS/multiclass_cnn.py" --fusion {args.fusion}')
        sys.exit(1)

    model = tf.keras.models.load_model(mpath)
    df = load_fusion_frame(args.fusion)
    x_all, y_all = frame_to_xy(df)

    end = min(args.start + args.steps, len(x_all))
    if args.start >= len(x_all):
        print("Start index out of range.")
        sys.exit(1)

    print(f"Streaming {args.fusion} rows [{args.start}, {end}) — delay {args.delay_ms} ms")
    correct = 0
    latencies: list[float] = []

    for i in range(args.start, end):
        t0 = time.perf_counter()
        batch = reshape_for_cnn(x_all[i])
        proba = model.predict(batch, verbose=0)[0]
        pred = int(np.argmax(proba))
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        true_id = int(y_all[i])
        if pred == true_id:
            correct += 1
        true_name = ATTACK_DISPLAY_NAMES.get(true_id, ATTACK_CLASS_NAMES.get(true_id, "?"))
        pred_name = ATTACK_DISPLAY_NAMES.get(pred, ATTACK_CLASS_NAMES.get(pred, "?"))
        conf = float(proba[pred])
        match_tag = "OK" if pred == true_id else "MISMATCH"
        info = get_attack_info(pred)
        if is_intrusion(pred, conf):
            action = info.recommended_actions[0]
            print(
                f"row={i:6d}  ALERT [{info.severity}] {pred_name}  conf={conf:.2%}  "
                f"-> {action}  ({dt_ms:.0f}ms, {match_tag})"
            )
        else:
            print(
                f"row={i:6d}  true={true_name:28s}  pred={pred_name:28s}  "
                f"conf={conf:.2%}  {dt_ms:6.1f}ms  {match_tag}"
            )
        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)

    n = end - args.start
    print("---")
    print(f"Samples: {n}  Accuracy: {correct/n:.1%}  Avg latency: {np.mean(latencies):.1f} ms")


if __name__ == "__main__":
    main()
