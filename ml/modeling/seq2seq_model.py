from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
        )
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self._debug_logged = False

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.transpose(1, 2)
        down_lengths = torch.ceil(lengths.float() / 4.0).long()
        t_prime = x.shape[1]
        out_lens = torch.clamp(down_lengths, min=1, max=t_prime)
        if not self._debug_logged:
            print(
                f"Debug: seq2seq downsample factor=4 T'={t_prime} lengths min={out_lens.min().item()} max={out_lens.max().item()}"
            )
            self._debug_logged = True
        out_lens_cpu = out_lens.to("cpu", dtype=torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, out_lens_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return enc_out, out_lens


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, enc_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attn_proj = nn.Linear(enc_dim, hidden_dim, bias=False)
        self.gru = nn.GRU(
            hidden_dim + enc_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.out = nn.Linear(hidden_dim + enc_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_lens: torch.Tensor,
        decoder_in: torch.Tensor,
    ) -> torch.Tensor:
        # encoder_out: [B, T, enc_dim], decoder_in: [B, U]
        batch_size, max_t, enc_dim = encoder_out.shape
        max_u = decoder_in.shape[1]
        enc_proj = self.attn_proj(encoder_out)
        embedded = self.embedding(decoder_in)

        outputs = []
        hidden = None
        for t in range(max_u):
            emb_t = embedded[:, t : t + 1, :]
            if hidden is None:
                query = emb_t
            else:
                query = hidden[-1].unsqueeze(1)
            scores = torch.bmm(enc_proj, query.transpose(1, 2)).squeeze(2)
            mask = torch.arange(max_t, device=encoder_lens.device).unsqueeze(0) >= encoder_lens.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=1).unsqueeze(1)
            context = torch.bmm(attn, encoder_out)
            gru_in = torch.cat([emb_t, context], dim=2)
            out_t, hidden = self.gru(gru_in, hidden)
            logits = self.out(torch.cat([out_t, context], dim=2))
            outputs.append(logits)

        return torch.cat(outputs, dim=1)


class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.encoder = Seq2SeqEncoder(input_dim, hidden_dim)
        self.decoder = AttentionDecoder(vocab_size, hidden_dim, hidden_dim * 2)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out, enc_lens = self.encoder(x, x_lens)
        logits = self.decoder(enc_out, enc_lens, y_in)
        return logits, enc_lens
