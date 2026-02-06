from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from ml.modeling.dataset import Seq2SeqDataset, collate_seq2seq, load_manifest
from ml.modeling.seq2seq_model import Seq2SeqModel
from ml.modeling.text import Vocab, decode_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate seq2seq baseline.")
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
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
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
        default=os.path.join("reports", "eval_seq2seq"),
        help="Output directory.",
    )
    return parser.parse_args()


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
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        candidate_best = os.path.join(checkpoint_path, "best.pt")
        candidate_last = os.path.join(checkpoint_path, "checkpoint.pt")
        if os.path.exists(candidate_best):
            checkpoint_path = candidate_best
        elif os.path.exists(candidate_last):
            checkpoint_path = candidate_last
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

    dataset = Seq2SeqDataset(samples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_seq2seq(batch, vocab.pad_id),
    )

    device = torch.device(args.device)
    model = Seq2SeqModel(input_dim=126, hidden_dim=256, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cer_total = 0.0
    wer_total = 0.0
    count = 0
    sample_lines: List[str] = []
    with torch.no_grad():
        for xs, x_lens, y_in, y_out, ids, texts in loader:
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

    avg_cer = cer_total / max(1, count)
    avg_wer = wer_total / max(1, count)

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics_eval.json")
    samples_path = os.path.join(args.out, "samples.txt")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"cer": avg_cer, "wer": avg_wer, "samples": count}, handle, ensure_ascii=False, indent=2)
    with open(samples_path, "w", encoding="utf-8-sig") as handle:
        handle.write("\n".join(sample_lines))

    print(f"CER={avg_cer:.4f} WER={avg_wer:.4f} samples={count}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote samples to {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
