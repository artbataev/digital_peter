import argparse
import logging
import multiprocessing
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as utils_rnn
from ctcdecode import CTCBeamDecoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from digital_peter import models
from digital_peter.data import OcrDataBatch, DigitalPeterDataset, DigitalPeterEvalDataset, collate_fn, OcrDataItem
from digital_peter.learning import OcrLearner
from digital_peter.logging_utils import setup_logger
from digital_peter.models.utils import update_bn_stats
from digital_peter.text import TextEncoder, get_chars_from_file, calc_metrics

DATA_DIR = Path(__file__).parent / "data"


def write_utt2hyp(utt2hyp: Dict[str, str], dir_path: Path):
    dir_path.mkdir(exist_ok=True)
    for uttid, hyp in utt2hyp.items():
        with open(dir_path / f"{uttid}.txt", "w", encoding="utf-8") as f:
            print(hyp, file=f)


def get_utt2hyp(model, loader, parl_decoder, encoder):
    model.eval()
    utt2hyp: Dict[str, str] = dict()
    with torch.no_grad():
        ocr_data_batch: OcrDataBatch
        for batch_idx, ocr_data_batch in enumerate(tqdm(loader)):
            images = ocr_data_batch.images.cuda()
            image_lengths = ocr_data_batch.image_lengths.cuda()

            logits, logits_lengths = model(images, image_lengths)
            log_logits = F.log_softmax(logits, dim=-1)

            beam_results, _, _, out_lens = parl_decoder.decode(
                log_logits.transpose(0, 1).detach(),
                seq_lens=logits_lengths)

            for i, uttid in enumerate(ocr_data_batch.keys):
                hyp_len = out_lens[i][0]
                hyp_encoded = beam_results[i, 0, :hyp_len]
                hyp = encoder.decode(hyp_encoded.numpy().tolist()).strip()
                utt2hyp[uttid] = hyp
    return utt2hyp


def get_utt2hyp_merged(model, dataset, parl_decoder, encoder):
    model.eval()
    utt2hyp: Dict[str, str] = dict()
    item: OcrDataItem
    merged_items = defaultdict(list)
    for i in range(len(dataset)):
        item = dataset[i]
        key_base, line = item.key.rsplit("_", maxsplit=1)
        merged_items[key_base].append(item)
    with torch.no_grad():
        for key_base, items in tqdm(merged_items.items()):
            items.sort(key=lambda item: item.key.rsplit("_", maxsplit=1)[1])
            images = []
            initial_images_lengths = []
            for item in items:
                images.append(item.img)  # CHW
                initial_images_lengths.append(item.img.shape[-1])
            initial_images_lengths = torch.LongTensor(initial_images_lengths)
            images = torch.cat(images, dim=-1)
            images_length = torch.LongTensor([images.shape[-1]])
            images = images.cuda().unsqueeze(0)
            images_length = images_length.cuda()
            logits_merged, _ = model(images, images_length)
            logits_merged = F.log_softmax(logits_merged, dim=-1)
            logits_merged = logits_merged.squeeze(1).cpu()  # LxC
            logits = []
            logits_lengths = initial_images_lengths // 4
            start_i = 0
            for cur_len in logits_lengths.numpy().tolist():
                logits.append(logits_merged[start_i:start_i + cur_len])
                start_i += cur_len
            log_logits = utils_rnn.pad_sequence(logits, batch_first=True)
            beam_results, _, _, out_lens = parl_decoder.decode(
                log_logits,
                seq_lens=logits_lengths)
            for i, item in enumerate(items):
                uttid = item.key
                hyp_len = out_lens[i][0]
                hyp_encoded = beam_results[i, 0, :hyp_len]
                hyp = encoder.decode(hyp_encoded.numpy().tolist()).strip()
                utt2hyp[uttid] = hyp
    return utt2hyp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", type=str)
    parser.add_argument("--from-ckp", type=str)
    parser.add_argument("--img-height", type=int, default=128)
    parser.add_argument("--bs", type=int, default=10, help="batch size")
    parser.add_argument("--lm", type=str, default="data/lang/lm_train_geval17_06_wbimaxent_0.8.gz")
    parser.add_argument("--lmwt", type=float, default=1.0)
    parser.add_argument("--wip", type=float, default=2.0)
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument("--test-img-dir", default="/data")
    parser.add_argument("--test-hyps-dir", default="/output")
    parser.add_argument("--adapt", action="store_true", help="update batchnorm stats using test data")
    parser.add_argument("--merged", action="store_true", help="use merged images for evaluation")
    args = parser.parse_args()

    setup_logger()
    log = logging.getLogger("evalscript")
    log.info(f"args: {args}")

    chars = get_chars_from_file(DATA_DIR / "chars_new.txt")
    encoder = TextEncoder(chars)
    num_outputs = len(encoder.id2char)
    log.info(f"num outputs: {num_outputs}")

    phones_list = encoder.id2char.copy()
    phones_list[phones_list.index(" ")] = "$"
    phones_list[phones_list.index("[")] = "P"
    phones_list[phones_list.index("]")] = "Q"
    num_processes = multiprocessing.cpu_count() or 12  # can be zero
    parl_decoder = CTCBeamDecoder(
        phones_list,
        model_path=args.lm,
        alpha=args.lmwt,
        beta=args.wip,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=num_processes,
        blank_id=0,
        log_probs_input=True
    )

    model: nn.Module = getattr(models, args.model)(num_outputs=num_outputs)
    model = model.cuda()
    model.load_state_dict(torch.load(args.from_ckp, map_location="cuda"))

    if args.eval_mode:
        test_data = DigitalPeterEvalDataset(Path(args.test_img_dir),
                                            img_height=args.img_height, image_len_divisible_by=4)
        loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)
        if args.adapt:
            update_bn_stats(model, loader)
        utt2hyp = get_utt2hyp(model, loader, parl_decoder, encoder)
        write_utt2hyp(utt2hyp, Path(args.test_hyps_dir))
    else:
        with open(DATA_DIR / "val_uttids_set.pkl", "rb") as f:
            val_uttids = pickle.load(f)
        val_data = DigitalPeterDataset(DATA_DIR / "train", val_uttids,
                                       encoder,
                                       img_height=args.img_height, image_len_divisible_by=4,
                                       verbose=False, training=False)
        log.info(f"data: {len(val_data)}")

        loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)
        criterion = nn.CTCLoss(blank=0, reduction="none")
        if args.adapt:
            update_bn_stats(model, loader)
        if args.merged:
            utt2ref = dict()
            for item in val_data:
                utt2ref[item.key] = item.text
            utt2hyp = get_utt2hyp_merged(model, val_data, parl_decoder, encoder)
            calc_metrics(utt2hyp, utt2ref)
        else:
            # utt2hyp = get_utt2hyp(model, loader, parl_decoder, encoder)
            learner = OcrLearner(model, None, criterion, None, loader, encoder, parl_decoder=parl_decoder)
            learner.val_model(greedy=False)


if __name__ == "__main__":
    main()
