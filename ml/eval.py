from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        default=os.path.join("reports", "eval_metrics.json"),
        help="Metrics output path.",
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


def _greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, vocab: Vocab) -> List[str]:
    preds = torch.argmax(log_probs, dim=-1)
    decoded: List[str] = []
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


def main() -> int:
    args = _parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = Vocab(chars=ckpt["vocab"])

    samples = load_manifest(args.manifest, args.kp_dir, limit=args.limit)
    if not samples:
        print("No samples available for evaluation.")
        return 1

    dataset = KeypointTextDataset(samples, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device(args.device)
    model = BiLSTMCTC(input_dim=126, hidden_dim=256, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cer_total = 0.0
    wer_total = 0.0
    count = 0
    with torch.no_grad():
        for xs, x_lens, _, _, _, texts in loader:
            xs = xs.to(device)
            x_lens = x_lens.to(device)
            log_probs = model(xs, x_lens)
            decoded = _greedy_decode(log_probs, x_lens, vocab)
            for pred, gt in zip(decoded, texts):
                cer_total += _cer(pred, gt)
                wer_total += _wer(pred, gt)
                count += 1

    avg_cer = cer_total / max(1, count)
    avg_wer = wer_total / max(1, count)

    metrics = {"cer": avg_cer, "wer": avg_wer, "samples": count}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(f"CER={avg_cer:.4f} WER={avg_wer:.4f} samples={count}")
    print(f"Wrote metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
