from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from shutil import copyfile
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.modeling.dataset import (
    KeypointAugmentConfig,
    Seq2SeqDataset,
    collate_seq2seq,
    load_manifest,
)
from ml.modeling.seq2seq_model import Seq2SeqModel
from ml.modeling.text import Vocab, build_vocab, decode_ids, encode_text

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train seq2seq baseline.")
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
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--tf_start", type=float, default=1.0)
    parser.add_argument("--tf_end", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature_dropout", type=float, default=0.10)
    parser.add_argument("--frame_dropout", type=float, default=0.05)
    parser.add_argument("--input_noise_std", type=float, default=0.01)
    parser.add_argument("--time_mask_prob", type=float, default=0.20)
    parser.add_argument("--time_mask_max_ratio", type=float, default=0.20)
    parser.add_argument("--time_mask_num", type=int, default=2)
    parser.add_argument("--freq_penalty", type=float, default=0.0)
    parser.add_argument("--best_alpha", type=float, default=0.30)
    parser.add_argument("--best_beta", type=float, default=0.20)
    parser.add_argument("--best_decode_limit", type=int, default=20)
    parser.add_argument("--gate_top1", type=float, default=0.60)
    parser.add_argument("--gate_unique", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _split_samples(samples: List, seed: int = 42, val_ratio: float = 0.1) -> Tuple[List, List]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_size = max(1, int(len(samples) * val_ratio)) if samples else 0
    val_idx = set(indices[:val_size])
    train_samples = [samples[i] for i in indices if i not in val_idx]
    val_samples = [samples[i] for i in indices if i in val_idx]
    return train_samples, val_samples


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
    model: Seq2SeqModel,
    x: torch.Tensor,
    x_lens: torch.Tensor,
    vocab: Vocab,
    max_len: int,
) -> List[str]:
    device = x.device
    batch = x.shape[0]
    enc_out, enc_lens = model.encoder(x, x_lens)
    y_tokens = torch.full((batch, 1), vocab.bos_id, dtype=torch.long, device=device)
    outputs: List[str] = ["" for _ in range(batch)]
    finished = [False] * batch

    for _ in range(max_len):
        logits = model.decoder(enc_out, enc_lens, y_tokens)
        next_ids = torch.argmax(logits[:, -1, :], dim=-1)
        y_tokens = torch.cat([y_tokens, next_ids.unsqueeze(1)], dim=1)
        for i, idx in enumerate(next_ids.tolist()):
            if finished[i]:
                continue
            if idx == vocab.eos_id:
                finished[i] = True
                continue
            outputs[i] += decode_ids(vocab, [idx])
    return outputs


def main() -> int:
    args = _parse_args()
    _set_seed(args.seed)

    augment_config = KeypointAugmentConfig(
        feature_dropout=args.feature_dropout,
        frame_dropout=args.frame_dropout,
        input_noise_std=args.input_noise_std,
        time_mask_prob=args.time_mask_prob,
        time_mask_max_ratio=args.time_mask_max_ratio,
        time_mask_num=args.time_mask_num,
    )
    print(
        "Augment config (train only):",
        f"feature_dropout={augment_config.feature_dropout:.3f}",
        f"frame_dropout={augment_config.frame_dropout:.3f}",
        f"input_noise_std={augment_config.input_noise_std:.3f}",
        f"time_mask_prob={augment_config.time_mask_prob:.3f}",
        f"time_mask_max_ratio={augment_config.time_mask_max_ratio:.3f}",
        f"time_mask_num={augment_config.time_mask_num}",
        f"seed={args.seed}",
    )

    samples = load_manifest(args.manifest, args.kp_dir, limit=args.limit)
    if not samples:
        print("No samples available for training.")
        return 1
    print(f"First GT text: {samples[0].text}")
    print(f"First GT repr: {samples[0].text!r}")

    train_samples, val_samples = _split_samples(samples, seed=args.seed, val_ratio=0.1)
    if not train_samples or not val_samples:
        print("Not enough data after split.")
        return 1

    vocab = build_vocab(sample.text for sample in train_samples)
    print(f"Vocab size (incl blank): {len(vocab)}")
    freq = torch.zeros(len(vocab), dtype=torch.float32)
    for sample in train_samples:
        for tok_id in encode_text(vocab, sample.text):
            if tok_id == vocab.blank_id:
                continue
            freq[tok_id] += 1.0
    freq[vocab.pad_id] = 0.0
    freq[vocab.bos_id] = 0.0
    freq[vocab.eos_id] = 0.0
    freq_max = float(freq.max().item())
    if freq_max > 0.0:
        freq = freq / freq_max
    else:
        freq = torch.zeros_like(freq)
    freq = freq * freq

    augment_rng = torch.Generator()
    augment_rng.manual_seed(args.seed + 1)
    train_ds = Seq2SeqDataset(train_samples, vocab, augment_config, augment_rng)
    val_ds = Seq2SeqDataset(val_samples, vocab)

    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_seq2seq(batch, vocab.pad_id),
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_seq2seq(batch, vocab.pad_id),
    )

    device = torch.device(args.device)
    model = Seq2SeqModel(
        input_dim=126, hidden_dim=256, vocab_size=len(vocab), dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_id, label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {
        "train_loss": [],
        "val_cer": [],
        "val_wer": [],
        "best_epoch": None,
        "best_val_cer": None,
        "best_score": None,
    }
    best_val_cer = None
    best_cer_epoch = None
    best_cer_ckpt_path = os.path.join(args.out_dir, "cer_best.pt")
    best_score = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        if args.epochs > 1:
            ratio = (epoch - 1) / (args.epochs - 1)
        else:
            ratio = 1.0
        tf_ratio = args.tf_start + (args.tf_end - args.tf_start) * ratio
        model.train()
        total_loss = 0.0
        masked_ratio_total = 0.0
        masked_ratio_count = 0
        freq_penalty = freq.to(device) * args.freq_penalty if args.freq_penalty > 0 else None
        for xs, x_lens, y_in, y_out, time_mask_ratios, _, _ in train_loader:
            xs = xs.to(device)
            x_lens = x_lens.to(device)
            y_in = y_in.to(device)
            y_out = y_out.to(device)
            masked_ratio_total += float(time_mask_ratios.sum().item())
            masked_ratio_count += int(time_mask_ratios.numel())

            if tf_ratio >= 0.999:
                logits, _ = model(xs, x_lens, y_in)
                if freq_penalty is not None:
                    logits = logits - freq_penalty
            else:
                enc_out, enc_lens = model.encoder(xs, x_lens)
                batch_size, max_u = y_in.shape
                y_tokens = y_in[:, :1]
                step_logits = []
                for t in range(max_u):
                    logits_t = model.decoder(enc_out, enc_lens, y_tokens)
                    last_logits = logits_t[:, -1, :]
                    if freq_penalty is not None:
                        step_logits.append(last_logits - freq_penalty)
                    else:
                        step_logits.append(last_logits)
                    if t + 1 < max_u:
                        use_gt = torch.rand(batch_size, device=device) < tf_ratio
                        pred_next = torch.argmax(last_logits, dim=-1)
                        gt_next = y_in[:, t + 1]
                        next_token = torch.where(use_gt, gt_next, pred_next)
                        y_tokens = torch.cat([y_tokens, next_token.unsqueeze(1)], dim=1)
                logits = torch.stack(step_logits, dim=1)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        avg_time_mask_ratio = masked_ratio_total / max(1, masked_ratio_count)
        metrics["train_loss"].append(avg_loss)

        model.eval()
        cer_total = 0.0
        wer_total = 0.0
        count = 0
        sample_lines: List[str] = []
        pred_texts: List[str] = []
        decode_limit = max(0, args.best_decode_limit)
        with torch.no_grad():
            for xs, x_lens, y_in, y_out, _time_mask_ratios, ids, texts in val_loader:
                xs = xs.to(device)
                x_lens = x_lens.to(device)
                max_len = y_in.shape[1] + 1
                preds = _greedy_decode(model, xs, x_lens, vocab, max_len)
                for sample_id, pred, gt in zip(ids, preds, texts):
                    cer_total += _cer(pred, gt)
                    wer_total += _wer(pred, gt)
                    count += 1
                    if len(sample_lines) < 20:
                        sample_lines.append(f"{sample_id}\tPRED: {pred}\tGT: {gt}")
                    if decode_limit > 0 and len(pred_texts) < decode_limit:
                        pred_texts.append(pred)

        avg_cer = cer_total / max(1, count)
        avg_wer = wer_total / max(1, count)
        metrics["val_cer"].append(avg_cer)
        metrics["val_wer"].append(avg_wer)

        unique_count = len(set(pred_texts))
        total_count = len(pred_texts)
        unique_ratio = (unique_count / total_count) if total_count > 0 else 0.0
        top1_pred, top1_count = ("", 0)
        if pred_texts:
            top1_pred, top1_count = Counter(pred_texts).most_common(1)[0]
        top1_ratio = (top1_count / total_count) if total_count > 0 else 0.0
        score = avg_cer + args.best_alpha * (1.0 - unique_ratio) + args.best_beta * top1_ratio

        if best_val_cer is None or avg_cer < best_val_cer:
            best_val_cer = avg_cer
            best_cer_epoch = epoch
            best_cer_ckpt = {"model_state": model.state_dict(), "vocab": vocab.chars}
            torch.save(best_cer_ckpt, best_cer_ckpt_path)

        passed_gate = top1_ratio < args.gate_top1 and unique_ratio > args.gate_unique
        status = "PASS" if passed_gate else "REJECT_COLLAPSE"
        improved = passed_gate and (best_score is None or score < best_score)
        if improved:
            best_score = score
            best_epoch = epoch
            best_ckpt = {"model_state": model.state_dict(), "vocab": vocab.chars}
            torch.save(best_ckpt, os.path.join(args.out_dir, "best.pt"))

        print(
            f"Epoch {epoch}: CER={avg_cer:.4f} unique={unique_count}/{total_count} "
            f"(ratio={unique_ratio:.3f}) top1={top1_count}/{total_count} "
            f"(ratio={top1_ratio:.3f}) score={score:.4f} status={status} "
            f"{'*best*' if improved else ''}"
        )

        with open(os.path.join(args.out_dir, "samples.txt"), "w", encoding="utf-8-sig") as handle:
            handle.write("\n".join(sample_lines))

    ckpt = {"model_state": model.state_dict(), "vocab": vocab.chars}
    torch.save(ckpt, os.path.join(args.out_dir, "checkpoint.pt"))
    selection_mode = "PASS_SCORE"
    if best_score is None:
        selection_mode = "FALLBACK_CER_ONLY"
        if os.path.exists(best_cer_ckpt_path):
            copyfile(best_cer_ckpt_path, os.path.join(args.out_dir, "best.pt"))
        best_epoch = best_cer_epoch
        print("FALLBACK_CER_ONLY: no epoch passed gate; using lowest CER checkpoint")
    print(f"Best selection mode: {selection_mode}")

    metrics["best_epoch"] = best_epoch
    metrics["best_val_cer"] = best_val_cer
    metrics["best_score"] = best_score
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(f"Saved checkpoint to {os.path.join(args.out_dir, 'checkpoint.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
