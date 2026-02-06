from __future__ import annotations

import torch
import math

from torch import nn


class BiLSTMCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int = 2,
        blank_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.blank_bias = blank_bias
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)
        self._debug_logged = False

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        down_lengths = torch.ceil(lengths.float() / 2.0).long()
        t_prime = x.shape[1]
        out_lens = torch.clamp(down_lengths, min=1, max=t_prime)
        if not self._debug_logged:
            print(
                f"Debug: downsampled T'={t_prime} lengths min={out_lens.min().item()} max={out_lens.max().item()}"
            )
            print(f"Debug: T'={t_prime} max_out_len={out_lens.max().item()}")
            self._debug_logged = True
        out_lens_cpu = out_lens.to("cpu", dtype=torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, out_lens_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(out)
        if self.training and self.blank_bias:
            logits[..., 0] -= float(self.blank_bias)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, out_lens
