from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

import torch
from torch.utils.data import DataLoader

from ml.modeling.dataset import KeypointTextDataset, collate_batch, load_manifest
from ml.modeling.model import BiLSTMCTC
from ml.modeling.text import Vocab, decode_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline CTC model.")
    parser.add_argument(
        "--manifest",
        default=os.path.join("data", "manifest", "manifest.jsonl"),
        help="Manifest path.",
    )
    parser.add_argument(
        "--kp_dir",
        default=os.path.join("data", "processed", "keypoints"),
        help="Keypoints directory.",
    )
    parser.add_argument("--checkpoint", help="Checkpoint path.")
    parser.add_argument("--ckpt", help="Alias for --checkpoint.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("reports", "eval"),
        help="Output directory (metrics_eval.json, samples.txt).",
    )
    parser.add_argument(
        "--out_dir",
        help="Alias for --out.",
    )
    return parser.parse_args()


def _edit_distance(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _cer(pred: str, gt: str) -> float:
    if not gt:
        return 1.0 if pred else 0.0
    return _edit_distance(list(pred), list(gt)) / max(1, len(gt))


def _wer(pred: str, gt: str) -> float:
    gt_tokens = gt.split()
    pred_tokens = pred.split()
    if not gt_tokens:
        return 1.0 if pred_tokens else 0.0
    return _edit_distance(pred_tokens, gt_tokens) / max(1, len(gt_tokens))


def _greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    vocab: Vocab,
    debug_blank: bool = False,
) -> List[str]:
    preds = torch.argmax(log_probs, dim=-1)
    decoded: List[str] = []
    if debug_blank:
        blank_id = vocab.blank_id
        total = 0
        blanks = 0
        for i in range(preds.shape[0]):
            seq = preds[i, : lengths[i]].tolist()
            total += len(seq)
            blanks += sum(1 for idx in seq if idx == blank_id)
        blank_ratio = (blanks / total) if total else 0.0
        print(f"Debug: blank_ratio={blank_ratio:.3f}")
    for i in range(preds.shape[0]):
        seq = preds[i, : lengths[i]].tolist()
        collapsed: List[int] = []
        prev = None
        for idx in seq:
            if idx != prev:
                collapsed.append(idx)
            prev = idx
        decoded.append(decode_ids(vocab, collapsed))
    return decoded


def _split_samples(samples: List, seed: int = 42, val_ratio: float = 0.1) -> tuple[List, List]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_size = max(1, int(len(samples) * val_ratio)) if samples else 0
    val_idx = set(indices[:val_size])
    train_samples = [samples[i] for i in indices if i not in val_idx]
    val_samples = [samples[i] for i in indices if i in val_idx]
    return train_samples, val_samples


def main() -> int:
    args = _parse_args()
    checkpoint_path = args.checkpoint or args.ckpt
    if not checkpoint_path:
        print("ERROR: checkpoint not provided.")
        return 1
    if os.path.isdir(checkpoint_path):
        candidate_best = os.path.join(checkpoint_path, "best.pt")
        candidate_last = os.path.join(checkpoint_path, "checkpoint.pt")
        if os.path.exists(candidate_best):
            checkpoint_path = candidate_best
        elif os.path.exists(candidate_last):
            checkpoint_path = candidate_last
        else:
            print(f"ERROR: no checkpoint found in directory: {checkpoint_path}")
            return 1
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return 1

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    vocab = Vocab(chars=ckpt["vocab"])

    samples = load_manifest(args.manifest, args.kp_dir, limit=args.limit)
    if not samples:
        print("No samples available for evaluation.")
        return 1
    train_samples, val_samples = _split_samples(samples, seed=42, val_ratio=0.1)
    if args.split == "train":
        samples = train_samples
    elif args.split == "val":
        samples = val_samples

    dataset = KeypointTextDataset(samples, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device(args.device)
    model = BiLSTMCTC(input_dim=126, hidden_dim=256, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cer_total = 0.0
    wer_total = 0.0
    count = 0
    sample_lines: List[str] = []
    with torch.no_grad():
        for xs, x_lens, _, _, _, texts in loader:
            xs = xs.to(device)
            x_lens = x_lens.to(device)
            log_probs, out_lens = model(xs, x_lens)
            decoded = _greedy_decode(log_probs, out_lens, vocab, debug_blank=(count == 0))
            for pred, gt in zip(decoded, texts):
                cer_total += _cer(pred, gt)
                wer_total += _wer(pred, gt)
                count += 1
                if len(sample_lines) < 20:
                    pred_snip = pred[:20]
                    gt_snip = gt[:20]
                    sample_lines.append(
                        f"len_pred={len(pred)} len_gt={len(gt)} pred='{pred_snip}' gt='{gt_snip}'"
                    )

    avg_cer = cer_total / max(1, count)
    avg_wer = wer_total / max(1, count)

    out_dir = args.out_dir or args.out
    if os.path.splitext(out_dir)[1] in {".json", ".txt"}:
        out_dir = os.path.dirname(out_dir)
    if not out_dir:
        out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics_eval.json")
    samples_path = os.path.join(out_dir, "samples.txt")
    metrics = {"cer": avg_cer, "wer": avg_wer, "samples": count}
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with open(samples_path, "w", encoding="utf-8-sig") as handle:
        handle.write("\n".join(sample_lines))

    print(f"CER={avg_cer:.4f} WER={avg_wer:.4f} samples={count}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote samples to {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
