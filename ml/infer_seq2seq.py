from __future__ import annotations

import argparse
import json
import os
from typing import Any, List

import numpy as np
import torch

from ml.modeling.seq2seq_model import Seq2SeqModel
from ml.modeling.text import Vocab, decode_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seq2seq inference for a single sample.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--manifest",
        default=os.path.join("data", "manifest", "manifest.jsonl"),
        help="Manifest path (required if using --id).",
    )
    parser.add_argument(
        "--kp_dir",
        default=os.path.join("data", "processed", "keypoints"),
        help="Keypoints directory (required if using --id).",
    )
    parser.add_argument("--id", help="Sample id from manifest.")
    parser.add_argument("--npz", help="Direct path to keypoints NPZ.")
    parser.add_argument("--device", default="cpu")
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
    return parser.parse_args()


def _find_npz_for_id(manifest_path: str, kp_dir: str, sample_id: str) -> str | None:
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            if record.get("id") == sample_id:
                candidate = os.path.join(kp_dir, f"{sample_id}.npz")
                return candidate
    return None


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

    npz_path = args.npz
    if args.id:
        npz_path = _find_npz_for_id(args.manifest, args.kp_dir, args.id)
        if not npz_path:
            print(f"ERROR: unable to resolve NPZ for id={args.id}")
            return 1

    if not npz_path or not os.path.exists(npz_path):
        print("ERROR: NPZ path not found.")
        return 1

    data = np.load(npz_path, allow_pickle=True)
    keypoints = data["keypoints"].astype(np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 126:
        print(f"ERROR: unexpected keypoints shape: {keypoints.shape}")
        return 1

    device = torch.device(args.device)
    model = Seq2SeqModel(input_dim=126, hidden_dim=256, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.from_numpy(keypoints).unsqueeze(0).to(device)
    if args.zero_input:
        print("DEBUG: zero_input=ON")
        x = torch.zeros_like(x)
    lengths = torch.tensor([keypoints.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        if args.beam_size > 1:
            decoded = _beam_search_decode(
                model,
                x,
                lengths,
                vocab,
                args.beam_size,
                args.max_len,
                args.length_penalty,
                args.no_repeat_ngram,
                args.max_repeat_token,
                args.min_len,
            )[0]
        else:
            decoded = _greedy_decode(
                model,
                x,
                lengths,
                vocab,
                max_len=args.max_len,
                no_repeat_ngram=args.no_repeat_ngram,
                max_repeat_token=args.max_repeat_token,
                min_len=args.min_len,
            )[0]

    print(f"NPZ: {npz_path}")
    print(f"Decoded: {decoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
