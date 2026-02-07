from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize collapsed predictions from samples.txt.")
    parser.add_argument(
        "--samples",
        default=Path("reports") / "eval_seq2seq" / "samples.txt",
        help="Path to samples.txt (ID<TAB>PRED<TAB>GT).",
    )
    parser.add_argument("--top_n", type=int, default=5, help="Top N predictions to report.")
    parser.add_argument(
        "--id_limit",
        type=int,
        default=20,
        help="Max IDs to show for the most-collapsed prediction.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"ERROR: samples file not found: {samples_path}")
        return 1

    preds: list[str] = []
    ids_by_pred: dict[str, list[str]] = defaultdict(list)
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            sample_id, pred, _gt = parts[0], parts[1], parts[2]
            preds.append(pred)
            ids_by_pred[pred].append(sample_id)

    if not preds:
        print("No predictions found in samples file.")
        return 0

    total = len(preds)
    counts = Counter(preds)
    top = counts.most_common(args.top_n)
    top_pred, top_count = top[0]
    top_ratio = top_count / total if total else 0.0

    print(f"Most frequent pred: '{top_pred}' count={top_count} ratio={top_ratio:.3f}")
    print("Top predictions:")
    for pred, count in top:
        ratio = count / total if total else 0.0
        print(f"- '{pred}' count={count} ratio={ratio:.3f}")
    print("IDs for most frequent pred:")
    for sample_id in ids_by_pred[top_pred][: args.id_limit]:
        print(f"- {sample_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
