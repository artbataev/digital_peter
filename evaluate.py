import argparse
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from ctcdecode import CTCBeamDecoder
from torch.utils.data import DataLoader
import kaldi_io

from digital_peter import models
from digital_peter.data import DigitalPeterDataset, collate_fn
from digital_peter.learning import OcrLearner
from digital_peter.logging_utils import setup_logger
from digital_peter.text import TextEncoder, get_chars_from_file

DATA_DIR = Path(__file__).parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=10, help="batch size")
    parser.add_argument("--model", default="base", type=str)
    parser.add_argument("--from-ckp", type=str)
    args = parser.parse_args()

    setup_logger()
    log = logging.getLogger("trainscript")
    log.info(f"args: {args}")

    chars = get_chars_from_file(DATA_DIR / "chars_new.txt")
    encoder = TextEncoder(chars)

    with open(DATA_DIR / "val_uttids_set.pkl", "rb") as f:
        val_uttids = pickle.load(f)
    val_data = DigitalPeterDataset(DATA_DIR / "train", val_uttids, encoder, image_len_divisible_by=4,
                                   verbose=False, training=False)
    log.info(f"data: {len(val_data)}")
    num_outputs = len(encoder.id2char)
    log.info(f"num outputs: {num_outputs}")

    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    phones_list = encoder.id2char.copy()
    phones_list[phones_list.index(" ")] = "$"
    phones_list[phones_list.index("[")] = "P"
    phones_list[phones_list.index("]")] = "Q"
    parl_decoder = CTCBeamDecoder(
        phones_list,
        model_path=f"{DATA_DIR / 'mixed666.gz'}",
        alpha=1.0,
        beta=2.0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=6,
        blank_id=0,
        log_probs_input=True
    )

    model: nn.Module = getattr(models, args.model)(num_outputs=num_outputs)
    model = model.cuda()
    model.load_state_dict(torch.load(args.from_ckp, map_location="cuda"))
    criterion = nn.CTCLoss(blank=0, reduction="none")

    learner = OcrLearner(model, None, criterion, None, val_loader, encoder, parl_decoder=parl_decoder)

    # learner.val_model()
    # learner.val_model(greedy=False)
    utt2logits = learner.get_val_logits()
    with open("logits.ark", "wb") as f:
        for key, torch_logits in utt2logits.items():
            np_logits = torch_logits.numpy()
            kaldi_io.write_mat(f, np_logits, key=key)


if __name__ == "__main__":
    main()
