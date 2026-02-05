import argparse
import json
import math
import os
import sys
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


FEATURE_DIM = 21 * 3 * 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract MediaPipe Holistic hand keypoints from a manifest."
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join("data", "manifest", "manifest.jsonl"),
        help="Path to manifest.jsonl.",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("data", "processed", "keypoints"),
        help="Output directory for per-sample NPZ files.",
    )
    parser.add_argument(
        "--qc_out",
        default=os.path.join("reports", "keypoints_qc.json"),
        help="QC report output path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only N samples.")
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Optional resize width (keeps aspect ratio).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resize_frame(frame: np.ndarray, resize_width: int | None) -> np.ndarray:
    if not resize_width:
        return frame
    height, width = frame.shape[:2]
    if width == resize_width:
        return frame
    scale = resize_width / float(width)
    new_height = int(round(height * scale))
    return cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)


def _hand_landmarks_to_vec(hand_landmarks) -> np.ndarray:
    if hand_landmarks is None:
        return np.zeros((21, 3), dtype=np.float32)
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )
    if coords.shape != (21, 3):
        pad = np.zeros((21, 3), dtype=np.float32)
        pad[: coords.shape[0], : coords.shape[1]] = coords
        return pad
    return coords


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_manifest(path: str, limit: int | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def main() -> int:
    args = _parse_args()
    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    samples = _read_manifest(args.manifest, args.limit)
    if not samples:
        print("No samples found to process.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    _ensure_parent_dir(args.qc_out)

    processed_ok = 0
    failed = 0
    frames_list: list[int] = []
    missing_left_list: list[float] = []
    missing_right_list: list[float] = []
    summaries: list[dict[str, Any]] = []

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        for sample in tqdm(samples, desc="Extracting", unit="sample"):
            sample_id = sample.get("id")
            video_path = sample.get("video_path")
            start_sec = _safe_float(sample.get("start_sec"))
            end_sec = _safe_float(sample.get("end_sec"))

            if not sample_id or not video_path:
                print(f"WARNING: missing id or video_path in sample: {sample}", file=sys.stderr)
                failed += 1
                continue

            if start_sec is None or end_sec is None or end_sec <= start_sec:
                print(
                    f"WARNING: bad time range for {sample_id}: start={start_sec}, end={end_sec}",
                    file=sys.stderr,
                )
                failed += 1
                continue

            out_path = os.path.join(args.out_dir, f"{sample_id}.npz")
            if os.path.exists(out_path) and not args.overwrite:
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"WARNING: cannot open video for {sample_id}: {video_path}", file=sys.stderr)
                failed += 1
                continue

            fps_value = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_value) if fps_value and fps_value > 0 else None
            if fps is None:
                print(f"WARNING: invalid fps for {sample_id}: {video_path}", file=sys.stderr)
                cap.release()
                failed += 1
                continue

            start_frame = int(math.floor(start_sec * fps))
            end_frame = int(math.ceil(end_sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            keypoints: list[np.ndarray] = []
            missing_left = 0
            missing_right = 0

            frame_index = start_frame
            while frame_index < end_frame:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = _resize_frame(frame, args.resize_width)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                left_vec = _hand_landmarks_to_vec(results.left_hand_landmarks)
                right_vec = _hand_landmarks_to_vec(results.right_hand_landmarks)
                if results.left_hand_landmarks is None:
                    missing_left += 1
                if results.right_hand_landmarks is None:
                    missing_right += 1

                feature = np.concatenate([left_vec.flatten(), right_vec.flatten()]).astype(
                    np.float32
                )
                keypoints.append(feature)

                frame_index += 1

            cap.release()

            if not keypoints:
                print(f"WARNING: no frames extracted for {sample_id}", file=sys.stderr)
                failed += 1
                continue

            keypoints_arr = np.stack(keypoints).astype(np.float32)
            frame_count = int(keypoints_arr.shape[0])
            missing_left_rate = float(missing_left / frame_count) if frame_count else 0.0
            missing_right_rate = float(missing_right / frame_count) if frame_count else 0.0

            meta = {
                "id": sample_id,
                "video_path": video_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "fps": fps,
                "frame_count": frame_count,
                "feature_dim": FEATURE_DIM,
                "missing_left_rate": missing_left_rate,
                "missing_right_rate": missing_right_rate,
            }

            np.savez(out_path, keypoints=keypoints_arr, meta=meta)
            processed_ok += 1
            frames_list.append(frame_count)
            missing_left_list.append(missing_left_rate)
            missing_right_list.append(missing_right_rate)
            summaries.append(
                {
                    "id": sample_id,
                    "frames": frame_count,
                    "missing_left": missing_left_rate,
                    "missing_right": missing_right_rate,
                    "out_path": out_path.replace("\\", "/"),
                }
            )

    avg_frames = sum(frames_list) / processed_ok if processed_ok else 0.0
    avg_missing_left = sum(missing_left_list) / processed_ok if processed_ok else 0.0
    avg_missing_right = sum(missing_right_list) / processed_ok if processed_ok else 0.0

    worst_sorted = sorted(
        summaries,
        key=lambda item: max(item["missing_left"], item["missing_right"]),
        reverse=True,
    )
    summary_trimmed = summaries[:50] + worst_sorted[:20]

    qc = {
        "total_samples": len(samples),
        "processed_ok": processed_ok,
        "failed": failed,
        "avg_frames": avg_frames,
        "avg_missing_left": avg_missing_left,
        "avg_missing_right": avg_missing_right,
        "samples": summary_trimmed,
    }

    with open(args.qc_out, "w", encoding="utf-8") as qc_f:
        json.dump(qc, qc_f, ensure_ascii=False, indent=2)

    print(
        "QC:",
        f"total={len(samples)}",
        f"ok={processed_ok}",
        f"failed={failed}",
        f"avg_frames={avg_frames:.2f}",
        f"avg_missing_left={avg_missing_left:.3f}",
        f"avg_missing_right={avg_missing_right:.3f}",
    )
    print(f"Wrote QC report to {args.qc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
