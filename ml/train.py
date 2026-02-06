from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.modeling.dataset import KeypointTextDataset, collate_batch, load_manifest
from ml.modeling.model import BiLSTMCTC
from ml.modeling.text import Vocab, build_vocab, decode_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline CTC model.")
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
        "--out_dir",
        required=True,
        help="Output directory for checkpoints and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_samples(samples: List, seed: int = 42, val_ratio: float = 0.1) -> Tuple[List, List]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_size = max(1, int(len(samples) * val_ratio)) if samples else 0
    val_idx = set(indices[:val_size])
    train_samples = [samples[i] for i in indices if i not in val_idx]
    val_samples = [samples[i] for i in indices if i in val_idx]
    return train_samples, val_samples


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


def main() -> int:
    args = _parse_args()
    _set_seed(42)

    samples = load_manifest(args.manifest, args.kp_dir, limit=args.limit)
    if not samples:
        print("No samples available for training.")
        return 1

    train_samples, val_samples = _split_samples(samples, seed=42, val_ratio=0.1)
    if not train_samples or not val_samples:
        print("Not enough data after split.")
        return 1

    vocab = build_vocab(sample.text for sample in train_samples)
    print(f"Vocab size (incl blank): {len(vocab)}")

    train_ds = KeypointTextDataset(train_samples, vocab)
    val_ds = KeypointTextDataset(val_samples, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    device = torch.device(args.device)
    model = BiLSTMCTC(input_dim=126, hidden_dim=256, vocab_size=len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {"train_loss": [], "val_cer": [], "val_wer": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xs, x_lens, targets, y_lens, _, _ in train_loader:
            xs = xs.to(device)
            x_lens = x_lens.to(device)
            targets = targets.to(device)
            y_lens = y_lens.to(device)

            log_probs = model(xs, x_lens)
            log_probs_t = log_probs.transpose(0, 1)
            loss = criterion(log_probs_t, targets, x_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        metrics["train_loss"].append(avg_loss)

        model.eval()
        cer_total = 0.0
        wer_total = 0.0
        count = 0
        sample_lines: List[str] = []
        with torch.no_grad():
            for xs, x_lens, _, _, ids, texts in val_loader:
                xs = xs.to(device)
                x_lens = x_lens.to(device)
                log_probs = model(xs, x_lens)
                decoded = _greedy_decode(log_probs, x_lens, vocab)
                for sample_id, pred, gt in zip(ids, decoded, texts):
                    cer_total += _cer(pred, gt)
                    wer_total += _wer(pred, gt)
                    count += 1
                    if len(sample_lines) < 20:
                        sample_lines.append(f"{sample_id}\tPRED: {pred}\tGT: {gt}")

        avg_cer = cer_total / max(1, count)
        avg_wer = wer_total / max(1, count)
        metrics["val_cer"].append(avg_cer)
        metrics["val_wer"].append(avg_wer)

        print(
            f"Epoch {epoch}/{args.epochs} - loss={avg_loss:.4f} val_CER={avg_cer:.4f} val_WER={avg_wer:.4f}"
        )

        with open(os.path.join(args.out_dir, "samples.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(sample_lines))

    ckpt = {
        "model_state": model.state_dict(),
        "vocab": vocab.chars,
    }
    torch.save(ckpt, os.path.join(args.out_dir, "checkpoint.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(f"Saved checkpoint to {os.path.join(args.out_dir, 'checkpoint.pt')}")
    print(f"Saved metrics to {os.path.join(args.out_dir, 'metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
