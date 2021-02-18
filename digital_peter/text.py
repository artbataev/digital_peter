import itertools
import logging
import pickle
from collections import Counter
from typing import Set, List, Dict

import editdistance


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
    chars = {" "}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            char = line.strip()
            if char:
                chars.add(char)
    return chars


def calc_metrics(utt2hyp: Dict[str, str], utt2ref: Dict[str, str]):
    error_chars = 0
    total_chars = 0
    error_words = 0
    total_words = 0
    error_sentences = 0
    log = logging.getLogger(__name__)

    for i, (uttid, hyp) in enumerate(utt2hyp.items()):
        ref = utt2ref[uttid]
        total_chars += len(ref)
        total_words += len(ref.split())
        error_chars += editdistance.eval(hyp, ref)
        error_words += editdistance.eval(hyp.split(), ref.split())
        if ref != hyp:
            error_sentences += 1
        if i < 20:
            log.info(f"{uttid} ref: {ref}")
            log.info(f"{uttid} hyp: {hyp}")
    cer = error_chars / total_chars
    wer = error_words / total_words
    num_sentences = len(utt2hyp)
    sentence_accuracy = (num_sentences - error_sentences) / num_sentences
    log.info(f"CER: {cer * 100:.3f}%, WER: {wer * 100:.3f}%, SACC: {sentence_accuracy * 100:.3f}%")
    return cer, wer
