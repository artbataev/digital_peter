from typing import List

import torch
from tqdm.auto import tqdm

from digital_peter.data import OcrDataBatch


def update_bn_stats(model, loader):
    model.train()
    with torch.no_grad():
        ocr_data_batch: OcrDataBatch
        for batch_idx, ocr_data_batch in enumerate(tqdm(loader)):
            images = ocr_data_batch.images.cuda()
            image_lengths = ocr_data_batch.image_lengths.cuda()
            _ = model(images, image_lengths)


def make_avg_model_from_checkpoints(checkpoints: List[str]):
    weights = [
        torch.load(ckp, map_location="cpu") for ckp in checkpoints
    ]
    avg_weights = weights[0]
    num_models = float(len(checkpoints))
    for key in avg_weights:
        for other_weights in weights[1:]:
            avg_weights[key] += other_weights[key]
        avg_weights[key] /= num_models
    return avg_weights
