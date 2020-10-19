import logging
from pathlib import Path
from typing import Union, Set

import cv2
import numpy as np
import torch
import torch.nn.utils.rnn as utils_rnn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from digital_peter.image import process_image
from digital_peter.text import TextEncoder


class DigitalPeterDataset(Dataset):
    def __init__(self,
                 base_dir: Union[Path, str],
                 uttids: Set[str],
                 encoder: TextEncoder,
                 image_len_divisible_by=1,
                 verbose=True):
        super().__init__()
        base_dir = Path(base_dir)
        self.trans_dir = base_dir / "words"
        self.image_dir = base_dir / "images"
        self.images = []
        self.texts = []
        self.encoded_texts = []
        self.keys = []
        self.encoder = encoder
        log = logging.getLogger(__name__)

        for uttid in tqdm(sorted(uttids)):
            imagepath = self.image_dir / f"{uttid}.jpg"
            textpath = self.trans_dir / f"{uttid}.txt"
            with open(textpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            try:
                encoded_text = self.encoder.encode(text)
            except KeyError:
                if verbose:
                    log.info(f"Skipping {uttid}, can't transcribe: {text}")
                continue
            self.keys.append(uttid)
            self.texts.append(text)
            self.encoded_texts.append(torch.LongTensor(encoded_text))
            img = cv2.imread(f"{imagepath}")
            img = process_image(img)
            width = img.shape[1]
            if width % image_len_divisible_by != 0:
                right_add = np.ones([img.shape[0], image_len_divisible_by - width % image_len_divisible_by, 3],
                                    dtype=np.float32)
                img = np.concatenate((img, right_add), axis=1)

            self.images.append(torch.FloatTensor(img.transpose(2, 0, 1)))  # HxWxC -> CxHxW

    def __getitem__(self, index):
        return self.images[index], self.texts[index], self.encoded_texts[index], len(self.texts[index])

    def __len__(self):
        return len(self.texts)


def collate_fn(items):
    items.sort(key=lambda item: -item[0].shape[-1])  # sort by image width
    image_lengths = torch.LongTensor([item[0].shape[-1] for item in items])
    images = utils_rnn.pad_sequence([item[0].transpose(0, -1) for item in items],
                                    batch_first=False).permute(1, 3, 2, 0)  # CxHxW -> WxHxC -> WxBxHxC -> BxCxHxW
    texts = [item[1] for item in items]
    encoded_texts = utils_rnn.pad_sequence([item[2] for item in items], batch_first=True)
    text_lengths = torch.LongTensor([item[2].shape[0] for item in items])
    return images, texts, encoded_texts, image_lengths, text_lengths
