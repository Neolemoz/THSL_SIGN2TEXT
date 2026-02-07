from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .text import Vocab, encode_text, encode_text_seq2seq, normalize_text


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    npz_path: str


@dataclass(frozen=True)
class KeypointAugmentConfig:
    feature_dropout: float = 0.0
    frame_dropout: float = 0.0
    input_noise_std: float = 0.0


def _apply_keypoint_augment(
    x: torch.Tensor,
    rng: torch.Generator,
    config: KeypointAugmentConfig,
) -> torch.Tensor:
    if (
        config.feature_dropout <= 0.0
        and config.frame_dropout <= 0.0
        and config.input_noise_std <= 0.0
    ):
        return x
    if x.numel() == 0:
        return x

    x = x.clone()
    if config.feature_dropout > 0.0 and x.shape[1] > 0:
        feature_mask = torch.rand(x.shape[1], generator=rng) < config.feature_dropout
        if feature_mask.any().item():
            x[:, feature_mask] = 0.0

    if config.frame_dropout > 0.0 and x.shape[0] > 0:
        frame_mask = torch.rand(x.shape[0], generator=rng) < config.frame_dropout
        if frame_mask.any().item():
            x[frame_mask] = 0.0

    if config.input_noise_std > 0.0:
        nonzero = x != 0
        if nonzero.any().item():
            noise = torch.randn(
                x.shape, generator=rng, device=x.device, dtype=x.dtype
            ) * config.input_noise_std
            x = x + noise * nonzero
    return x


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
    def __init__(
        self,
        samples: List[Sample],
        vocab: Vocab,
        augment_config: KeypointAugmentConfig | None = None,
        augment_rng: torch.Generator | None = None,
    ) -> None:
        self.samples = samples
        self.vocab = vocab
        self.augment_config = augment_config
        if augment_config is not None and augment_rng is None:
            augment_rng = torch.Generator()
            augment_rng.manual_seed(torch.initial_seed())
        self.augment_rng = augment_rng

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        data = np.load(sample.npz_path, allow_pickle=True)
        keypoints = data["keypoints"].astype(np.float32)
        x = torch.from_numpy(keypoints)
        if self.augment_config is not None and self.augment_rng is not None:
            x = _apply_keypoint_augment(x, self.augment_rng, self.augment_config)
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


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        vocab: Vocab,
        augment_config: KeypointAugmentConfig | None = None,
        augment_rng: torch.Generator | None = None,
    ) -> None:
        self.samples = samples
        self.vocab = vocab
        self.augment_config = augment_config
        if augment_config is not None and augment_rng is None:
            augment_rng = torch.Generator()
            augment_rng.manual_seed(torch.initial_seed())
        self.augment_rng = augment_rng

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        data = np.load(sample.npz_path, allow_pickle=True)
        keypoints = data["keypoints"].astype(np.float32)
        x = torch.from_numpy(keypoints)
        if self.augment_config is not None and self.augment_rng is not None:
            x = _apply_keypoint_augment(x, self.augment_rng, self.augment_config)
        y_ids = encode_text_seq2seq(self.vocab, sample.text)
        y = torch.tensor(y_ids, dtype=torch.long)
        return {
            "id": sample.sample_id,
            "x": x,
            "x_len": x.shape[0],
            "y": y,
            "y_len": y.shape[0],
            "text": sample.text,
        }


def collate_seq2seq(batch: List[dict[str, Any]], pad_id: int) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[str],
    List[str],
]:
    if not batch:
        raise ValueError("Empty batch")
    batch_sorted = sorted(batch, key=lambda item: item["x_len"], reverse=True)
    max_x = batch_sorted[0]["x_len"]
    feature_dim = batch_sorted[0]["x"].shape[1]
    max_y = max(item["y_len"] for item in batch_sorted)

    xs = torch.zeros((len(batch_sorted), max_x, feature_dim), dtype=torch.float32)
    x_lens = torch.zeros((len(batch_sorted),), dtype=torch.long)
    y_in = torch.full((len(batch_sorted), max_y - 1), pad_id, dtype=torch.long)
    y_out = torch.full((len(batch_sorted), max_y - 1), pad_id, dtype=torch.long)
    ids: List[str] = []
    texts: List[str] = []

    for i, item in enumerate(batch_sorted):
        x = item["x"]
        x_len = item["x_len"]
        y = item["y"]
        y_len = item["y_len"]
        xs[i, :x_len] = x
        x_lens[i] = x_len
        if y_len > 1:
            y_in[i, : y_len - 1] = y[:-1]
            y_out[i, : y_len - 1] = y[1:]
        ids.append(item["id"])
        texts.append(item["text"])

    return xs, x_lens, y_in, y_out, ids, texts
