from __future__ import annotations

import argparse
import json
import os
from typing import Any, List

import numpy as np
import torch

from ml.modeling.model import BiLSTMCTC
from ml.modeling.text import Vocab, decode_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single sample.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path.",
    )
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
    return parser.parse_args()


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


def main() -> int:
    args = _parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    ckpt = torch.load(args.checkpoint, map_location="cpu")
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
    model = BiLSTMCTC(input_dim=126, hidden_dim=256, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.from_numpy(keypoints).unsqueeze(0).to(device)
    lengths = torch.tensor([keypoints.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        log_probs = model(x, lengths)
        decoded = _greedy_decode(log_probs, lengths, vocab)[0]

    print(f"NPZ: {npz_path}")
    print(f"Decoded: {decoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
