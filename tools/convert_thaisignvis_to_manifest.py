import argparse
import json
import os
import sys
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ThaiSignVis subset to a manifest.jsonl."
    )
    parser.add_argument(
        "--root",
        default=os.path.join("data", "raw", "thaisignvis"),
        help="Root directory containing transcript_window_*.csv files.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "manifest", "manifest.jsonl"),
        help="Output manifest path.",
    )
    parser.add_argument(
        "--qc_out",
        default=os.path.join("reports", "thaisignvis_qc.json"),
        help="QC report output path.",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iter_transcript_windows(root: str) -> list[str]:
    matches: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith("transcript_window_") and filename.endswith(".csv"):
                matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def _window_id_from_name(path: str) -> str:
    name = os.path.basename(path)
    stem = name.replace("transcript_window_", "").replace(".csv", "")
    return stem


def _forward_slash(path: str) -> str:
    return path.replace("\\", "/")


def _pick_start_column(df: pd.DataFrame) -> str | None:
    for col in ("relative_strart", "relative_start", "start"):
        if col in df.columns:
            return col
    return None


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def main() -> int:
    args = _parse_args()
    transcript_paths = _iter_transcript_windows(args.root)
    windows_found = len(transcript_paths)

    if windows_found == 0:
        print(f"No transcript_window_*.csv files found under {args.root}.")

    skipped_empty_text = 0
    skipped_bad_time = 0
    skipped_missing_video = 0

    durations: list[float] = []
    text_lengths: list[int] = []
    samples_written = 0

    _ensure_parent_dir(args.out)
    _ensure_parent_dir(args.qc_out)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for transcript_path in transcript_paths:
            window_id = _window_id_from_name(transcript_path)
            video_name = f"process_video_{window_id}.mp4"
            video_path = os.path.join(os.path.dirname(transcript_path), video_name)
            if not os.path.exists(video_path):
                print(
                    f"WARNING: missing video for window {window_id}: {video_path}",
                    file=sys.stderr,
                )
                skipped_missing_video += 1
                continue

            df = pd.read_csv(transcript_path)
            start_col = _pick_start_column(df)
            if start_col is None:
                print(
                    f"WARNING: missing start column in {transcript_path}",
                    file=sys.stderr,
                )
                skipped_bad_time += len(df)
                continue

            rel_video_path = _forward_slash(os.path.relpath(video_path))

            for row_idx, row in df.iterrows():
                text_th = _row_text(row.get("text", "")).strip()
                if not text_th:
                    skipped_empty_text += 1
                    continue

                start_sec = _safe_float(row.get(start_col))
                duration = _safe_float(row.get("duration"))
                if start_sec is None or duration is None or duration <= 0:
                    skipped_bad_time += 1
                    continue

                sample = {
                    "id": f"TSV_{window_id}_{int(row_idx):06d}",
                    "video_path": rel_video_path,
                    "text_th": text_th,
                    "start_sec": start_sec,
                    "end_sec": start_sec + duration,
                    "duration": duration,
                    "window_id": window_id,
                    "source": "thaisignvis",
                }
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                samples_written += 1
                durations.append(duration)
                text_lengths.append(len(text_th))

            print(
                f"Processed window {window_id}: rows={len(df)} written={samples_written}"
            )

    if samples_written:
        avg_duration = sum(durations) / samples_written
        avg_text_len = sum(text_lengths) / samples_written
        min_duration = min(durations)
        max_duration = max(durations)
    else:
        avg_duration = 0.0
        avg_text_len = 0.0
        min_duration = 0.0
        max_duration = 0.0

    qc = {
        "samples_written": samples_written,
        "windows_found": windows_found,
        "skipped_empty_text": skipped_empty_text,
        "skipped_bad_time": skipped_bad_time,
        "skipped_missing_video": skipped_missing_video,
        "avg_duration": avg_duration,
        "avg_text_len": avg_text_len,
        "min_duration": min_duration,
        "max_duration": max_duration,
    }

    with open(args.qc_out, "w", encoding="utf-8") as qc_f:
        json.dump(qc, qc_f, ensure_ascii=False, indent=2)

    print(
        "QC:",
        f"samples_written={samples_written}",
        f"windows_found={windows_found}",
        f"skipped_empty_text={skipped_empty_text}",
        f"skipped_bad_time={skipped_bad_time}",
        f"skipped_missing_video={skipped_missing_video}",
        f"avg_duration={avg_duration:.3f}",
        f"avg_text_len={avg_text_len:.2f}",
        f"min_duration={min_duration:.3f}",
        f"max_duration={max_duration:.3f}",
    )
    print(f"Wrote manifest to {args.out}")
    print(f"Wrote QC report to {args.qc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
