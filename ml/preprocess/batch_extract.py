from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


FEATURE_DIM = 21 * 3 * 2
HAND_DIM = 21 * 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract MediaPipe hand keypoints from a manifest."
    )
    parser.add_argument(
        "--manifest",
        default=Path("data") / "manifest" / "manifest.jsonl",
        help="Path to manifest.jsonl.",
    )
    parser.add_argument(
        "--out_dir",
        default=Path("data") / "processed" / "keypoints",
        help="Output directory for per-sample NPZ files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files.")
    parser.add_argument("--limit", type=int, default=None, help="Process only N samples.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1).")
    return parser.parse_args()


def _read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _validate_npz(path: Path) -> tuple[bool, str]:
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        return False, f"load_error:{exc}"
    if "keypoints" not in data:
        return False, "missing_keypoints"
    keypoints = data["keypoints"]
    if keypoints.dtype != np.float32:
        return False, f"bad_dtype:{keypoints.dtype}"
    if keypoints.ndim != 2 or keypoints.shape[1] != FEATURE_DIM:
        return False, f"bad_shape:{keypoints.shape}"
    return True, "ok"


def _backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidate = path.with_suffix(path.suffix + ".bak")
    index = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f".bak{index}")
        index += 1
    candidate.write_bytes(path.read_bytes())
    return candidate


def _save_npz(path: Path, keypoints: np.ndarray, meta: dict[str, Any]) -> Path | None:
    temp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez(str(temp_path), keypoints=keypoints, meta=meta)
    backup = _backup_existing(path)
    temp_path.replace(path)
    return backup


def _extract_sample(
    sample: dict[str, Any],
    out_dir: Path,
    overwrite: bool,
    holistic: Any | None,
) -> dict[str, Any]:
    sample_id = sample.get("id")
    video_path = sample.get("video_path")
    if not sample_id or not video_path:
        return {"id": sample_id or "", "status": "failed", "reason": "missing_id_or_video"}

    out_path = out_dir / f"{sample_id}.npz"
    if out_path.exists() and not overwrite:
        valid, reason = _validate_npz(out_path)
        if valid:
            return {"id": sample_id, "status": "skipped_exists"}

    start_sec = _safe_float(sample.get("start_sec"))
    end_sec = _safe_float(sample.get("end_sec"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"id": sample_id, "status": "failed", "reason": f"open_video:{video_path}"}

    fps_value = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_value) if fps_value and fps_value > 0 else None
    if fps is None and (start_sec is not None or end_sec is not None):
        cap.release()
        return {"id": sample_id, "status": "failed", "reason": "invalid_fps"}

    start_frame = 0
    end_frame: int | None = None
    if fps is None:
        fps = 0.0
    if start_sec is not None:
        start_frame = int(math.floor(start_sec * fps))
    if end_sec is not None:
        end_frame = int(math.ceil(end_sec * fps))
    if end_frame is not None and end_frame <= start_frame:
        cap.release()
        return {"id": sample_id, "status": "failed", "reason": "bad_time_range"}

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    keypoints: list[np.ndarray] = []
    missing_left = 0
    missing_right = 0
    zero_frames = 0

    def _process_frames(holistic_obj) -> None:
        nonlocal missing_left, missing_right, zero_frames
        frame_index = start_frame
        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_obj.process(rgb)
            left_vec = _hand_landmarks_to_vec(results.left_hand_landmarks)
            right_vec = _hand_landmarks_to_vec(results.right_hand_landmarks)
            left_missing = results.left_hand_landmarks is None
            right_missing = results.right_hand_landmarks is None
            if left_missing:
                missing_left += 1
            if right_missing:
                missing_right += 1
            if left_missing and right_missing:
                zero_frames += 1
            feature = np.concatenate([left_vec.flatten(), right_vec.flatten()]).astype(
                np.float32
            )
            keypoints.append(feature)
            frame_index += 1

    if holistic is None:
        with mp.solutions.holistic.Holistic(
            model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic_obj:
            _process_frames(holistic_obj)
    else:
        _process_frames(holistic)

    cap.release()

    if not keypoints:
        return {"id": sample_id, "status": "failed", "reason": "no_frames"}

    keypoints_arr = np.stack(keypoints).astype(np.float32)
    frame_count = int(keypoints_arr.shape[0])
    missing_left_rate = float(missing_left / frame_count) if frame_count else 0.0
    missing_right_rate = float(missing_right / frame_count) if frame_count else 0.0
    zero_frame_ratio = float(zero_frames / frame_count) if frame_count else 0.0

    meta = {
        "id": sample_id,
        "video_path": str(video_path),
        "frames": frame_count,
        "fps": fps,
        "feature_dim": FEATURE_DIM,
        "missing_left": missing_left_rate,
        "missing_right": missing_right_rate,
        "zero_frame_ratio": zero_frame_ratio,
    }
    _save_npz(out_path, keypoints_arr, meta)
    return {
        "id": sample_id,
        "status": "processed",
        "frames": frame_count,
        "missing_left": missing_left_rate,
        "missing_right": missing_right_rate,
        "zero_frame_ratio": zero_frame_ratio,
    }


def _process_sample_worker(
    sample: dict[str, Any], out_dir: str, overwrite: bool
) -> dict[str, Any]:
    return _extract_sample(sample, Path(out_dir), overwrite, holistic=None)


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    samples = _read_manifest(manifest_path, args.limit)
    if not samples:
        print("No samples found to process.", file=sys.stderr)
        return 1

    total = len(samples)
    processed = 0
    skipped_exists = 0
    failed = 0

    tasks: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample.get("id")
        video_path = sample.get("video_path")
        if not sample_id or not video_path:
            failed += 1
            print(f"WARNING: missing id or video_path in sample: {sample}", file=sys.stderr)
            continue
        out_path = out_dir / f"{sample_id}.npz"
        if out_path.exists() and not args.overwrite:
            valid, reason = _validate_npz(out_path)
            if valid:
                skipped_exists += 1
                continue
            print(
                f"WARNING: invalid existing npz for {sample_id} ({reason}); regenerating",
                file=sys.stderr,
            )
        tasks.append(sample)

    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_sample_worker, sample, str(out_dir), args.overwrite): sample
                for sample in tasks
            }
            total_tasks = len(futures)
            done = 0
            for future in as_completed(futures):
                done += 1
                result = future.result()
                status = result.get("status")
                sample_id = result.get("id", "")
                if status == "processed":
                    processed += 1
                    print(f"[{done}/{total_tasks}] processed {sample_id}")
                elif status == "skipped_exists":
                    skipped_exists += 1
                    print(f"[{done}/{total_tasks}] skipped {sample_id}")
                else:
                    failed += 1
                    reason = result.get("reason", "unknown")
                    print(
                        f"WARNING: failed {sample_id}: {reason}",
                        file=sys.stderr,
                    )
    else:
        with mp.solutions.holistic.Holistic(
            model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            total_tasks = len(tasks)
            for idx, sample in enumerate(tasks, start=1):
                sample_id = sample.get("id", "")
                print(f"[{idx}/{total_tasks}] processing {sample_id}")
                result = _extract_sample(sample, out_dir, args.overwrite, holistic)
                status = result.get("status")
                if status == "processed":
                    processed += 1
                    print(f"[{idx}/{total_tasks}] processed {sample_id}")
                elif status == "skipped_exists":
                    skipped_exists += 1
                    print(f"[{idx}/{total_tasks}] skipped {sample_id}")
                else:
                    failed += 1
                    reason = result.get("reason", "unknown")
                    print(
                        f"WARNING: failed {sample_id}: {reason}",
                        file=sys.stderr,
                    )

    print(
        "Batch extract:",
        f"total={total}",
        f"processed={processed}",
        f"skipped_exists={skipped_exists}",
        f"failed={failed}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
