import argparse
from typing import Dict
from digital_peter.text import calc_metrics
from digital_peter.logging_utils import setup_logger


def read_utt2text(filename) -> Dict[str, str]:
    utt2text = dict()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            uttid, text = line.strip().split(maxsplit=1)
            utt2text[uttid] = " ".join(text.split())

    return utt2text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps")
    parser.add_argument("refs")
    args = parser.parse_args()

    setup_logger()

    utt2hyp = read_utt2text(args.hyps)
    utt2ref = read_utt2text(args.refs)
    print(calc_metrics(utt2hyp, utt2ref))
