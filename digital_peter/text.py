import itertools
from typing import Set, List


class TextEncoder:
    def __init__(self, allowed_chars: Set[str], use_unk=False):
        self.allowed_chars = allowed_chars
        self.id2char = ["<blank>"] + sorted(allowed_chars)
        self.use_unk = use_unk
        if use_unk:
            self.id2char.append("<unk>")
        self.unk_idx = len(self.id2char) - 1 if use_unk else -1
        self.char2id = dict(zip(self.id2char, range(len(self.id2char))))

    def encode(self, text: str):
        if self.use_unk:
            return [self.char2id[c] if c in self.allowed_chars else self.unk_idx for c in text]
        return [self.char2id[c] for c in text]

    def decode(self, sequence: List[int]):
        return "".join(self.id2char[i] for i in sequence if i != self.unk_idx)

    def decode_unk(self, sequence: List[int]):
        return "".join(self.id2char[i] for i in sequence)

    def decode_ctc(self, sequence: List[int]):
        return self.decode([i for i, _ in itertools.groupby(sequence) if i != 0])