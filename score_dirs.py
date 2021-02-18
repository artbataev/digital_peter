import argparse
from pathlib import Path
from typing import Dict

from digital_peter.logging_utils import setup_logger
from digital_peter.text import calc_metrics


def read_utt2text_dir(dirname) -> Dict[str, str]:
    utt2text = dict()
    for filepath in Path(dirname).glob("*.txt"):
        uttid = filepath.stem
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        utt2text[uttid] = " ".join(text.split())

    return utt2text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps_dir")
    parser.add_argument("refs_dir")
    args = parser.parse_args()

    setup_logger()

    utt2hyp = read_utt2text_dir(args.hyps_dir)
    utt2ref = read_utt2text_dir(args.refs_dir)
    print(calc_metrics(utt2hyp, utt2ref))
