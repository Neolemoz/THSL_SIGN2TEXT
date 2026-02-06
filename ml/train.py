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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--blank_bias", type=float, default=0.0)
    parser.add_argument("--blank_bias_start", type=float, default=1.0)
    parser.add_argument("--blank_bias_end", type=float, default=0.0)
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


def _blank_ratio(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> float:
    preds = torch.argmax(log_probs, dim=-1)
    total = 0
    blanks = 0
    for i in range(preds.shape[0]):
        seq = preds[i, : lengths[i]].tolist()
        total += len(seq)
        blanks += sum(1 for idx in seq if idx == blank_id)
    return (blanks / total) if total else 0.0


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
    print(f"First GT text: {samples[0].text}")
    print(f"First GT repr: {samples[0].text!r}")

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
    model = BiLSTMCTC(
        input_dim=126,
        hidden_dim=256,
        vocab_size=len(vocab),
        blank_bias=args.blank_bias,
    ).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    warmup_steps = max(1, int(0.05 * total_steps))
    step_count = 0

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {
        "train_loss": [],
        "val_cer": [],
        "val_wer": [],
        "best_epoch": None,
        "best_val_cer": None,
    }
    debug_logged = False
    best_val_cer = None
    best_epoch = None
    blank_streak = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.epochs > 1:
            ratio = (epoch - 1) / (args.epochs - 1)
        else:
            ratio = 1.0
        scheduled_blank_bias = args.blank_bias_start + (
            args.blank_bias_end - args.blank_bias_start
        ) * ratio
        total_loss = 0.0
        for xs, x_lens, targets, y_lens, _, _ in train_loader:
            xs = xs.to(device)
            x_lens = x_lens.to(device)
            targets = targets.to(device)
            y_lens = y_lens.to(device)

            log_probs, out_lens = model(xs, x_lens, blank_bias=scheduled_blank_bias)
            log_probs_t = log_probs.transpose(0, 1)
            if not debug_logged:
                print(f"Debug: log_probs_t shape={tuple(log_probs_t.shape)} (T,N,C)")
                print(
                    f"Debug: input_lengths min={x_lens.min().item()} max={x_lens.max().item()}"
                )
                print(
                    f"Debug: target_lengths min={y_lens.min().item()} max={y_lens.max().item()}"
                )
                print(f"Debug: blank_id={vocab.blank_id}")
                debug_logged = True
            loss = criterion(log_probs_t, targets, out_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            step_count += 1
            warmup_scale = min(1.0, step_count / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup_scale
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
                log_probs, out_lens = model(xs, x_lens, blank_bias=0.0)
                decoded = _greedy_decode(log_probs, out_lens, vocab)
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

        train_cer_total = 0.0
        train_count = 0
        blank_ratios: List[float] = []
        with torch.no_grad():
            for batch_idx, (xs, x_lens, _, _, _, texts) in enumerate(train_loader):
                if batch_idx >= 8:
                    break
                xs = xs.to(device)
                x_lens = x_lens.to(device)
                log_probs, out_lens = model(xs, x_lens, blank_bias=0.0)
                blank_ratios.append(_blank_ratio(log_probs, out_lens, vocab.blank_id))
                decoded = _greedy_decode(
                    log_probs, out_lens, vocab, debug_blank=(batch_idx == 0)
                )
                for pred, gt in zip(decoded, texts):
                    train_cer_total += _cer(pred, gt)
                    train_count += 1
        train_cer = train_cer_total / max(1, train_count)
        avg_blank_ratio = sum(blank_ratios) / max(1, len(blank_ratios))
        if avg_blank_ratio < 0.05 or avg_blank_ratio > 0.95:
            print(
                f"Warning: blank_ratio={avg_blank_ratio:.3f} suggests tuning --blank_bias"
            )

        improved = best_val_cer is None or avg_cer < best_val_cer
        if improved:
            best_val_cer = avg_cer
            best_epoch = epoch
            best_ckpt = {
                "model_state": model.state_dict(),
                "vocab": vocab.chars,
            }
            torch.save(best_ckpt, os.path.join(args.out_dir, "best.pt"))

        print(
            f"Epoch {epoch}/{args.epochs} - loss={avg_loss:.4f} train_CER={train_cer:.4f} "
            f"val_CER={avg_cer:.4f} blank_ratio={avg_blank_ratio:.3f} "
            f"blank_bias={scheduled_blank_bias:.3f} {'*best*' if improved else ''}"
        )

        if avg_blank_ratio > 0.98:
            blank_streak += 1
        else:
            blank_streak = 0
        if blank_streak >= 2:
            print("Warning: blank_ratio > 0.98 for 2 epochs; stopping early.")
            break

        with open(os.path.join(args.out_dir, "samples.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(sample_lines))

    ckpt = {
        "model_state": model.state_dict(),
        "vocab": vocab.chars,
    }
    torch.save(ckpt, os.path.join(args.out_dir, "checkpoint.pt"))
    metrics["best_epoch"] = best_epoch
    metrics["best_val_cer"] = best_val_cer
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(f"Saved checkpoint to {os.path.join(args.out_dir, 'checkpoint.pt')}")
    print(f"Saved metrics to {os.path.join(args.out_dir, 'metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
