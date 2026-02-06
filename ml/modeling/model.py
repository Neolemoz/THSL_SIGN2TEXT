from __future__ import annotations

import torch
from torch import nn


class BiLSTMCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int = 2,
        blank_bias: float = 0.5,
    ) -> None:
        super().__init__()
        self.blank_bias = blank_bias
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(out)
        if self.training and self.blank_bias:
            logits[..., 0] -= float(self.blank_bias)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs
