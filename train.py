import argparse
import logging
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ctcdecode import CTCBeamDecoder
from torch.utils.data import DataLoader

from digital_peter import models
from digital_peter.data import DigitalPeterDataset, collate_fn
from digital_peter.learning import OcrLearner
from digital_peter.logging_utils import setup_logger
from digital_peter.text import TextEncoder, get_chars_from_file

DATA_DIR = Path(__file__).parent / "data"


def set_seed():
    random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    np.random.seed(111)


def get_exp_dir(args, num_chars: int) -> Path:
    model_str = f"{args.model}--{num_chars}"
    opt_str = f"ep-{args.start_ep}to{args.epochs}_lr-{args.init_lr}to{args.final_lr}_bs-{args.bs}" \
              f"_optim-{args.optim}-wd{args.wd}"
    exp_dir = f"{model_str}/{opt_str}"
    exp_dir = Path(args.exp_dir) / exp_dir
    return exp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--start-ep", type=int, default=0)
    parser.add_argument("--bs", type=int, default=10, help="batch size")
    parser.add_argument("--optim", type=str, choices={"adam", "adabelief", "sgd"}, default="adam")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--model", default="base", type=str, help="model name, from models module")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--from-ckp", type=str, default="")
    parser.add_argument("--exp-dir", type=str, default="exp")
    parser.add_argument("--force", default=False, action="store_true", help="ingore existing dir")
    args = parser.parse_args()

    set_seed()
    chars = get_chars_from_file(DATA_DIR / "chars_new.txt")

    exp_dir = get_exp_dir(args, len(chars))
    exp_dir.mkdir(exist_ok=args.force, parents=True)
    setup_logger(exp_dir)
    log = logging.getLogger("trainscript")
    log.info(f"args: {args}")

    encoder = TextEncoder(chars)
    assert len(chars) == 62

    with open(DATA_DIR / "train_uttids_set.pkl", "rb") as f:
        train_uttids = pickle.load(f)
    with open(DATA_DIR / "val_uttids_set.pkl", "rb") as f:
        val_uttids = pickle.load(f)
    train_data = DigitalPeterDataset(DATA_DIR / "train", train_uttids, encoder, image_len_divisible_by=4,
                                     verbose=False, training=True)
    val_data = DigitalPeterDataset(DATA_DIR / "train", val_uttids, encoder, image_len_divisible_by=4,
                                   verbose=False, training=False)
    log.info(f"data: {len(train_data), len(val_data)}")
    num_outputs = len(encoder.id2char)
    log.info(f"num outputs: {num_outputs}")

    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    phones_list = encoder.id2char.copy()
    phones_list[phones_list.index(" ")] = "$"
    parl_decoder = CTCBeamDecoder(
        phones_list,
        model_path=f"{DATA_DIR / 'phone_lm.gz'}",
        alpha=1.0,
        beta=2.0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=6,
        blank_id=0,
        log_probs_input=True
    )

    init_lr = args.init_lr
    final_lr = args.final_lr
    num_epochs = args.epochs

    model: nn.Module = getattr(models, args.model)(num_outputs=num_outputs)
    model = model.cuda()
    if args.from_ckp:
        model.load_state_dict(torch.load(args.from_ckp, map_location="cuda"))
    criterion = nn.CTCLoss(blank=0, reduction="none")
    if args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=args.wd)
    elif args.optim == "adabelief":
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(model.parameters(), lr=init_lr, eps=1e-10, betas=(0.9, 0.999), weight_decouple=False,
                              rectify=False, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        raise Exception("unknown optimizer")
    learner = OcrLearner(model, optimizer, criterion, train_loader, val_loader, encoder, parl_decoder=parl_decoder)

    reduce_lr = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda epoch: math.exp(math.log(final_lr / init_lr) * epoch / num_epochs))
    # best_loss = float("infinity")
    best_loss = learner.val_model()
    torch.save(learner.model.state_dict(), exp_dir / "model_best.pt")
    try:
        for i_epoch in range(args.start_ep, num_epochs):
            # if i_epoch == 0:
            #     train_data.shuffle_buckets(args.bs, shuffle_parts=False)  # pseudo sortagrad
            # else:
            #     train_data.shuffle_buckets(args.bs, shuffle_parts=True)
            log.info("=" * 50)
            log.info(f"epoch: {i_epoch + 1}")
            learner.train_model()
            cur_loss = learner.val_model()
            reduce_lr.step()
            if best_loss < cur_loss:
                log.info(f"not improved {best_loss:.5f} < {cur_loss:.5f}")
                torch.save(learner.model.state_dict(), exp_dir / "model_last.pt")
            else:
                log.info(f"improved {best_loss:.5f} -> {cur_loss:.5f}")
                best_loss = cur_loss
                torch.save(learner.model.state_dict(), exp_dir / "model_best.pt")
    except KeyboardInterrupt:
        log.warning("training interruped")

    model.load_state_dict(torch.load(exp_dir / "model_best.pt"))
    learner.val_model()
    learner.val_model(greedy=False)


if __name__ == "__main__":
    main()
