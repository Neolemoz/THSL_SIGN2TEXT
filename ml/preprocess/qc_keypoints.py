from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_DIM = 21 * 3 * 2
HAND_DIM = 21 * 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC report for keypoint NPZ files.")
    parser.add_argument(
        "--manifest",
        default=Path("data") / "manifest" / "manifest.jsonl",
        help="Path to manifest.jsonl.",
    )
    parser.add_argument(
        "--kp_dir",
        default=Path("data") / "processed" / "keypoints",
        help="Directory with keypoint .npz files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only N samples.")
    parser.add_argument(
        "--out",
        default=Path("reports") / "keypoints_qc.json",
        help="Output QC JSON path.",
    )
    parser.add_argument(
        "--skip_missing_npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, missing_npz are excluded from failed_quality and failed_samples.txt.",
    )
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--max_missing_left", type=float, default=0.8)
    parser.add_argument("--max_missing_right", type=float, default=0.8)
    parser.add_argument("--max_zero_frame_ratio", type=float, default=0.9)
    parser.add_argument("--expected_dim", type=int, default=FEATURE_DIM)
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


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


def _compute_missing_ratios(keypoints: np.ndarray) -> tuple[float, float, float]:
    if keypoints.ndim != 2 or keypoints.shape[1] != FEATURE_DIM or keypoints.shape[0] == 0:
        return 0.0, 0.0, 0.0
    left = keypoints[:, :HAND_DIM]
    right = keypoints[:, HAND_DIM:]
    left_missing = np.all(left == 0, axis=1)
    right_missing = np.all(right == 0, axis=1)
    zero_frames = np.all(keypoints == 0, axis=1)
    total = float(keypoints.shape[0])
    return (
        float(left_missing.mean()) if total else 0.0,
        float(right_missing.mean()) if total else 0.0,
        float(zero_frames.mean()) if total else 0.0,
    )


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    kp_dir = Path(args.kp_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    samples = _read_manifest(manifest_path, args.limit)
    if not samples:
        print("No samples found to QC.", file=sys.stderr)
        return 1

    frames_list: list[int] = []
    missing_left_list: list[float] = []
    missing_right_list: list[float] = []
    zero_frame_list: list[float] = []
    failures: list[dict[str, Any]] = []
    quality_failures: list[dict[str, Any]] = []
    present = 0
    missing_npz = 0

    for sample in samples:
        sample_id = sample.get("id")
        if not sample_id:
            failures.append({"id": "", "reason": "missing_id"})
            continue
        npz_path = kp_dir / f"{sample_id}.npz"
        if not npz_path.exists():
            failures.append({"id": sample_id, "reason": "missing_npz"})
            missing_npz += 1
            continue
        present += 1
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as exc:
            failures.append({"id": sample_id, "reason": f"load_error:{exc}"})
            continue
        if "keypoints" not in data:
            failures.append({"id": sample_id, "reason": "missing_keypoints"})
            continue
        keypoints = data["keypoints"]
        if keypoints.ndim != 2:
            failures.append({"id": sample_id, "reason": f"bad_shape:{keypoints.shape}"})
            continue
        if keypoints.dtype != np.float32:
            failures.append({"id": sample_id, "reason": f"bad_dtype:{keypoints.dtype}"})
            continue

        frames = int(keypoints.shape[0])
        feature_dim = int(keypoints.shape[1])
        missing_left, missing_right, zero_frame_ratio = _compute_missing_ratios(keypoints)

        frames_list.append(frames)
        missing_left_list.append(missing_left)
        missing_right_list.append(missing_right)
        zero_frame_list.append(zero_frame_ratio)

        fail_reasons: list[str] = []
        if frames < args.min_frames:
            fail_reasons.append("frames_lt_min")
        if missing_left > args.max_missing_left:
            fail_reasons.append("missing_left_high")
        if missing_right > args.max_missing_right:
            fail_reasons.append("missing_right_high")
        if zero_frame_ratio > args.max_zero_frame_ratio:
            fail_reasons.append("zero_frame_ratio_high")
        if feature_dim != args.expected_dim:
            fail_reasons.append("feature_dim_mismatch")
        if fail_reasons:
            failure = {
                "id": sample_id,
                "reason": ",".join(fail_reasons),
                "frames": frames,
                "missing_left": missing_left,
                "missing_right": missing_right,
                "zero_frame_ratio": zero_frame_ratio,
                "feature_dim": feature_dim,
            }
            failures.append(failure)
            quality_failures.append(failure)
    failed_quality = len(quality_failures)
    if not args.skip_missing_npz:
        failed_quality += missing_npz
    ok_count = present - len(quality_failures)
    qc = {
        "total": len(samples),
        "present": present,
        "missing_npz": missing_npz,
        "ok": ok_count,
        "failed_quality": failed_quality,
        "frames": _stats([float(v) for v in frames_list]),
        "missing_left": _stats(missing_left_list),
        "missing_right": _stats(missing_right_list),
        "zero_frame_ratio": _stats(zero_frame_list),
        "thresholds": {
            "min_frames": args.min_frames,
            "max_missing_left": args.max_missing_left,
            "max_missing_right": args.max_missing_right,
            "max_zero_frame_ratio": args.max_zero_frame_ratio,
            "expected_dim": args.expected_dim,
        },
        "failures": failures,
    }

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(qc, handle, ensure_ascii=False, indent=2)

    failed_list_path = out_path.parent / "failed_samples.txt"
    with failed_list_path.open("w", encoding="utf-8") as handle:
        if args.skip_missing_npz:
            for fail in quality_failures:
                handle.write(f"{fail.get('id', '')}\t{fail.get('reason', '')}\n")
        else:
            for fail in failures:
                handle.write(f"{fail.get('id', '')}\t{fail.get('reason', '')}\n")

    avg_frames = qc["frames"]["mean"]
    avg_missing_left = qc["missing_left"]["mean"]
    avg_missing_right = qc["missing_right"]["mean"]
    print(
        "QC:",
        f"total={len(samples)}",
        f"present={present}",
        f"missing_npz={missing_npz}",
        f"ok={ok_count}",
        f"failed_quality={failed_quality}",
        f"avg_frames={avg_frames:.2f}",
        f"avg_missing_left={avg_missing_left:.3f}",
        f"avg_missing_right={avg_missing_right:.3f}",
    )
    print(f"Wrote QC report to {out_path}")
    print(f"Wrote failed list to {failed_list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
