from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .text import Vocab, encode_text, normalize_text


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    npz_path: str


def load_manifest(
    manifest_path: str,
    kp_dir: str,
    limit: int | None = None,
    verbose: bool = True,
) -> List[Sample]:
    samples: List[Sample] = []
    skipped_missing = 0
    skipped_empty = 0

    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if limit and len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            sample_id = record.get("id")
            text_th = record.get("text_th", "")
            if not sample_id:
                continue
            text_norm = normalize_text(text_th)
            if not text_norm:
                skipped_empty += 1
                continue
            npz_path = os.path.join(kp_dir, f"{sample_id}.npz")
            if not os.path.exists(npz_path):
                skipped_missing += 1
                continue
            samples.append(Sample(sample_id=sample_id, text=text_norm, npz_path=npz_path))

    if verbose:
        print(
            "Manifest load:",
            f"samples={len(samples)}",
            f"skipped_missing_npz={skipped_missing}",
            f"skipped_empty_text={skipped_empty}",
        )
    return samples


class KeypointTextDataset(Dataset):
    def __init__(self, samples: List[Sample], vocab: Vocab) -> None:
        self.samples = samples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        data = np.load(sample.npz_path, allow_pickle=True)
        keypoints = data["keypoints"].astype(np.float32)
        x = torch.from_numpy(keypoints)
        y_ids = encode_text(self.vocab, sample.text)
        y = torch.tensor(y_ids, dtype=torch.long)
        return {
            "id": sample.sample_id,
            "x": x,
            "x_len": x.shape[0],
            "y": y,
            "y_len": y.shape[0],
            "text": sample.text,
        }


def collate_batch(batch: List[dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    if not batch:
        raise ValueError("Empty batch")
    batch_sorted = sorted(batch, key=lambda item: item["x_len"], reverse=True)
    max_len = batch_sorted[0]["x_len"]
    feature_dim = batch_sorted[0]["x"].shape[1]

    xs = torch.zeros((len(batch_sorted), max_len, feature_dim), dtype=torch.float32)
    x_lens = torch.zeros((len(batch_sorted),), dtype=torch.long)
    targets: List[torch.Tensor] = []
    y_lens = torch.zeros((len(batch_sorted),), dtype=torch.long)
    ids: List[str] = []
    texts: List[str] = []

    for i, item in enumerate(batch_sorted):
        x = item["x"]
        length = item["x_len"]
        xs[i, :length] = x
        x_lens[i] = length
        targets.append(item["y"])
        y_lens[i] = item["y_len"]
        ids.append(item["id"])
        texts.append(item["text"])

    targets_cat = torch.cat(targets, dim=0) if targets else torch.tensor([], dtype=torch.long)
    return xs, x_lens, targets_cat, y_lens, ids, texts
