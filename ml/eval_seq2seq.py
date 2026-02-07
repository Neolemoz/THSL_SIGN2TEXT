from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from ml.modeling.dataset import Seq2SeqDataset, collate_seq2seq, load_manifest
from ml.modeling.seq2seq_model import Seq2SeqModel
from ml.modeling.text import Vocab, decode_ids

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


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
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.0,
        help="Length normalization factor for beam search.",
    )
    parser.add_argument("--len_penalty", type=float, default=None, help="Alias for --length_penalty.")
    parser.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="Minimum decoded token length before EOS is allowed.",
    )
    parser.add_argument("--no_repeat_ngram", type=int, default=2)
    parser.add_argument("--max_repeat_token", type=int, default=3)
    parser.add_argument("--zero_input", action="store_true")
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


def _get_banned_tokens(generated: List[int], no_repeat_ngram: int) -> set[int]:
    if no_repeat_ngram <= 0:
        return set()
    if len(generated) < no_repeat_ngram - 1:
        return set()
    if no_repeat_ngram == 1:
        return set(generated)
    prefix = tuple(generated[-(no_repeat_ngram - 1) :])
    ngrams: dict[tuple[int, ...], set[int]] = {}
    for i in range(len(generated) - no_repeat_ngram + 1):
        prev = tuple(generated[i : i + no_repeat_ngram - 1])
        nxt = generated[i + no_repeat_ngram - 1]
        ngrams.setdefault(prev, set()).add(nxt)
    return ngrams.get(prefix, set())


def _apply_constraints(
    log_probs: torch.Tensor,
    generated: List[int],
    no_repeat_ngram: int,
    max_repeat_token: int,
    min_len: int,
    eos_id: int,
) -> torch.Tensor:
    if min_len > 0:
        current_len = max(0, len(generated) - 1)
        if current_len < min_len:
            log_probs[eos_id] = float("-inf")
    if max_repeat_token > 0 and len(generated) >= max_repeat_token:
        last = generated[-1]
        if all(tok == last for tok in generated[-max_repeat_token:]):
            log_probs[last] = float("-inf")
    if no_repeat_ngram > 0:
        banned = _get_banned_tokens(generated, no_repeat_ngram)
        if banned:
            log_probs[list(banned)] = float("-inf")
    return log_probs


def _greedy_decode(
    model: Seq2SeqModel,
    x: torch.Tensor,
    x_lens: torch.Tensor,
    vocab: Vocab,
    max_len: int,
    no_repeat_ngram: int,
    max_repeat_token: int,
    min_len: int,
) -> List[str]:
    device = x.device
    batch = x.shape[0]
    enc_out, enc_lens = model.encoder(x, x_lens)
    y_tokens = torch.full((batch, 1), vocab.bos_id, dtype=torch.long, device=device)
    outputs: List[str] = ["" for _ in range(batch)]
    finished = [False] * batch
    histories: List[List[int]] = [[vocab.bos_id] for _ in range(batch)]

    for _ in range(max_len):
        logits = model.decoder(enc_out, enc_lens, y_tokens)
        next_ids = []
        for i in range(batch):
            log_probs = torch.log_softmax(logits[i, -1, :], dim=-1)
            log_probs = _apply_constraints(
                log_probs,
                histories[i],
                no_repeat_ngram,
                max_repeat_token,
                min_len,
                vocab.eos_id,
            )
            next_ids.append(int(torch.argmax(log_probs).item()))
        next_ids = torch.tensor(next_ids, device=device)
        y_tokens = torch.cat([y_tokens, next_ids.unsqueeze(1)], dim=1)
        for i, idx in enumerate(next_ids.tolist()):
            if finished[i]:
                continue
            histories[i].append(idx)
            if idx == vocab.eos_id:
                finished[i] = True
                continue
            outputs[i] += decode_ids(vocab, [idx])
    return outputs


def _beam_search_single(
    decoder,
    enc_out: torch.Tensor,
    enc_lens: torch.Tensor,
    vocab: Vocab,
    beam_size: int,
    max_len: int,
    length_penalty: float,
    no_repeat_ngram: int,
    max_repeat_token: int,
    min_len: int,
) -> str:
    device = enc_out.device
    hyps: List[tuple[List[int], float, bool]] = [([vocab.bos_id], 0.0, False)]

    for _ in range(max_len):
        candidates: List[tuple[List[int], float, bool]] = []
        for tokens, score, finished in hyps:
            if finished:
                candidates.append((tokens, score, True))
                continue
            y = torch.tensor(tokens, device=device).unsqueeze(0)
            logits = decoder(enc_out, enc_lens, y)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            log_probs = _apply_constraints(
                log_probs,
                tokens,
                no_repeat_ngram,
                max_repeat_token,
                min_len,
                vocab.eos_id,
            )
            topk = torch.topk(log_probs, beam_size)
            for idx, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_tokens = tokens + [idx]
                candidates.append((new_tokens, score + lp, idx == vocab.eos_id))

        hyps = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
        if all(done for _, _, done in hyps):
            break

    def final_score(item: tuple[List[int], float, bool]) -> float:
        tokens, score, _ = item
        length = max(1, len(tokens) - 1)
        return score / (length ** length_penalty) if length_penalty > 0 else score

    best = max(hyps, key=final_score)
    return decode_ids(vocab, best[0])


def _beam_search_decode(
    model: Seq2SeqModel,
    x: torch.Tensor,
    x_lens: torch.Tensor,
    vocab: Vocab,
    beam_size: int,
    max_len: int,
    length_penalty: float,
    no_repeat_ngram: int,
    max_repeat_token: int,
    min_len: int,
) -> List[str]:
    enc_out, enc_lens = model.encoder(x, x_lens)
    outputs: List[str] = []
    for i in range(enc_out.shape[0]):
        outputs.append(
            _beam_search_single(
                model.decoder,
                enc_out[i : i + 1],
                enc_lens[i : i + 1],
                vocab,
                beam_size,
                max_len,
                length_penalty,
                no_repeat_ngram,
                max_repeat_token,
                min_len,
            )
        )
    return outputs


def main() -> int:
    args = _parse_args()
    print("DEBUG: eval_seq2seq starting")
    print(f"DEBUG: requested_limit={args.limit} batch_size={args.batch_size}")
    if args.len_penalty is not None:
        args.length_penalty = args.len_penalty
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

    samples = load_manifest(args.manifest, args.kp_dir, limit=None)
    if not samples:
        print("No samples available for evaluation.")
        return 1
    print(f"DEBUG: loaded_samples={len(samples)}")

    train_samples, val_samples = _split_samples(samples, seed=42, val_ratio=0.1)
    print(f"DEBUG: split_sizes train={len(train_samples)} val={len(val_samples)}")
    if args.split == "train":
        samples = train_samples
    elif args.split == "val":
        samples = val_samples
    print(f"DEBUG: samples_after_split={len(samples)} split={args.split}")

    dataset = Seq2SeqDataset(samples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_seq2seq(batch, vocab.pad_id),
    )
    print(f"DEBUG: dataset_len={len(dataset)} dataloader_len={len(loader)}")

    device = torch.device(args.device)
    model = Seq2SeqModel(input_dim=126, hidden_dim=256, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cer_total = 0.0
    wer_total = 0.0
    count = 0
    sample_lines: List[str] = []
    pred_texts: List[str] = []
    processed = 0
    requested_limit = args.limit
    exit_reason = "exhausted"
    try:
        with torch.no_grad():
            for batch_idx, (xs, x_lens, y_in, y_out, _time_mask_ratios, ids, texts) in enumerate(loader):
                if batch_idx < 5:
                    print(
                        f"DEBUG: batch_idx={batch_idx} batch_size={len(ids)} processed={processed}"
                    )
                xs = xs.to(device)
                x_lens = x_lens.to(device)
                if args.zero_input:
                    print("DEBUG: zero_input=ON")
                    xs = torch.zeros_like(xs)
                if args.beam_size > 1:
                    preds = _beam_search_decode(
                        model,
                        xs,
                        x_lens,
                        vocab,
                        args.beam_size,
                        args.max_len,
                        args.length_penalty,
                        args.no_repeat_ngram,
                        args.max_repeat_token,
                        args.min_len,
                    )
                else:
                    max_len = min(args.max_len, y_in.shape[1] + 1)
                    preds = _greedy_decode(
                        model,
                        xs,
                        x_lens,
                        vocab,
                        max_len,
                        args.no_repeat_ngram,
                        args.max_repeat_token,
                        args.min_len,
                    )
                for sample_id, pred, gt in zip(ids, preds, texts):
                    if requested_limit is not None and processed >= requested_limit:
                        exit_reason = "limit_reached"
                        break
                    cer_total += _cer(pred, gt)
                    wer_total += _wer(pred, gt)
                    count += 1
                    processed += 1
                    sample_lines.append(f"{sample_id}\tPRED: {pred}\tGT: {gt}")
                    pred_texts.append(pred)
                if requested_limit is not None and processed >= requested_limit:
                    exit_reason = "limit_reached"
                    break
    except Exception as exc:
        exit_reason = f"exception: {exc}"
        raise

    avg_cer = cer_total / max(1, count)
    avg_wer = wer_total / max(1, count)

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics_eval.json")
    samples_path = os.path.join(args.out, "samples.txt")
    unique_count = len(set(pred_texts))
    total_count = len(pred_texts)
    unique_ratio = (unique_count / total_count) if total_count > 0 else 0.0
    top5 = Counter(pred_texts).most_common(5)
    top1_pred, top1_count = ("", 0)
    if top5:
        top1_pred, top1_count = top5[0]
    top1_ratio = (top1_count / total_count) if total_count > 0 else 0.0
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cer": avg_cer,
                "wer": avg_wer,
                "samples": count,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "top1_pred": top1_pred,
                "top1_count": top1_count,
                "top1_ratio": top1_ratio,
                "top5_preds": top5,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with open(samples_path, "w", encoding="utf-8-sig") as handle:
        handle.write("\n".join(sample_lines))

    print(f"CER={avg_cer:.4f} WER={avg_wer:.4f} samples={count}")
    print(f"Diversity: unique={unique_count}/{total_count} top1='{top1_pred}' count={top1_count}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote samples to {samples_path}")
    print(f"EVAL: requested_limit={requested_limit} processed={processed} wrote={processed}")
    print(f"DEBUG: loop_exit={exit_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
