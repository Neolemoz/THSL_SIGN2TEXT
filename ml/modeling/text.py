from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def normalize_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # Collapse consecutive whitespace to single spaces.
    parts = text.split()
    return " ".join(parts)


@dataclass(frozen=True)
class Vocab:
    chars: List[str]

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def pad_id(self) -> int:
        return self.char_to_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.char_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.char_to_id(EOS_TOKEN)

    def __len__(self) -> int:
        return len(self.chars) + 1

    def char_to_id(self, ch: str) -> int:
        try:
            return self.chars.index(ch) + 1
        except ValueError:
            return self.blank_id

    def id_to_char(self, idx: int) -> str:
        if idx == self.blank_id:
            return ""
        real_idx = idx - 1
        if 0 <= real_idx < len(self.chars):
            return self.chars[real_idx]
        return ""

    def is_special(self, ch: str) -> bool:
        return ch in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}


def build_vocab(texts: Iterable[str]) -> Vocab:
    seen = set()
    ordered: List[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    for text in texts:
        normalized = normalize_text(text)
        for ch in normalized:
            if ch not in seen:
                seen.add(ch)
                ordered.append(ch)
    return Vocab(chars=ordered)


def encode_text(vocab: Vocab, text: str) -> List[int]:
    normalized = normalize_text(text)
    return [vocab.char_to_id(ch) for ch in normalized if ch]


def encode_text_seq2seq(vocab: Vocab, text: str) -> List[int]:
    normalized = normalize_text(text)
    tokens = [vocab.bos_id]
    tokens.extend(vocab.char_to_id(ch) for ch in normalized if ch)
    tokens.append(vocab.eos_id)
    return tokens


def decode_ids(vocab: Vocab, ids: Iterable[int]) -> str:
    chars: List[str] = []
    for idx in ids:
        if idx == vocab.blank_id:
            continue
        ch = vocab.id_to_char(idx)
        if not ch or vocab.is_special(ch):
            continue
        chars.append(ch)
    return "".join(chars)
