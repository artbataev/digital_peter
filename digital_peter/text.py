from typing import Set, List
import itertools


class TextEncoder:
    def __init__(self, allowed_chars: Set[str]):
        self.allowed_chars = allowed_chars
        self.id2char = ["<blank>"] + sorted(allowed_chars)
        self.char2id = dict(zip(self.id2char, range(len(self.id2char))))

    def encode(self, text: str):
        return [self.char2id[c] for c in text]

    def decode(self, sequence: List[int]):
        return "".join(self.id2char[i] for i in sequence)

    def decode_ctc(self, sequence: List[int]):
        return "".join(self.id2char[i] for i, _ in itertools.groupby(sequence) if i != 0)
