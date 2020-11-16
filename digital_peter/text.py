import itertools
from typing import Set, List
import pickle
from collections import Counter


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


def get_chars(counter_pkl_path, exclude_eng=False, min_char_freq=5) -> Set[str]:
    english = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w'}
    with open(counter_pkl_path, "rb") as f:
        chars_counter: Counter = pickle.load(f)
    chars = set()
    for char, cnt in chars_counter.items():
        if cnt >= min_char_freq:
            chars.add(char)
    if exclude_eng:
        chars -= english
    return chars


def get_chars_from_file(filename) -> Set[str]:
    chars = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line:
                chars.add(line)
    return chars
