import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Set, List

import cv2
import numpy as np
import torch
import torch.nn.utils.rnn as utils_rnn
from torch.utils.data import Dataset
from torchvision.transforms import functional
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from digital_peter.image import process_image
from digital_peter.text import TextEncoder


@dataclass
class OcrDataItem:
    key: str
    img: torch.Tensor
    text: str
    encoded_text: torch.Tensor
    encoded_text_len: int


@dataclass
class OcrDataBatch:
    keys: List[str]
    texts: List[str]
    images: torch.Tensor
    encoded_texts: torch.Tensor
    image_lengths: torch.Tensor
    text_lengths: torch.Tensor


def clean_text(text: str) -> str:
    # remove
    replacements = [
        ("і", "i"),
        ("c", "с"),
        ("a", "а"),
        ("⊗", "⊕"),
        ("lll", " "),
        ("–", " "),
        (")", " "),
        ("|", " "),
        ("×", "+"),
        ("k", "к"),
        ("ǂ", "+"),
    ]
    text = text.strip()
    for char_in, char_out in replacements:
        text = text.replace(char_in, char_out)
    text = " ".join(text.strip().split())  # remove additional spaces
    return text


def clean_text_file(file_in, file_out):
    keys_to_remove = {"20_16_0", "41_10_1", "47_20_5", "313_12_9"}
    with open(file_in, "r", encoding="utf-8") as f_in, open(file_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                uttid, text = line.strip().split(maxsplit=1)
            except ValueError:
                continue
            if uttid in keys_to_remove:
                continue
            text = clean_text(text)
            print(f"{uttid} {text}", file=f_out)


class DigitalPeterDataset(Dataset):
    def __init__(self,
                 base_dir: Union[Path, str],
                 uttids: Set[str],
                 encoder: TextEncoder,
                 image_len_divisible_by=4,
                 training=False,
                 verbose=True,
                 sort=False):
        super().__init__()
        base_dir = Path(base_dir)
        self.trans_dir = base_dir / "words"
        self.image_dir = base_dir / "images"
        self.images = []
        self.texts = []
        self.encoded_texts = []
        self.keys = []
        self.encoder = encoder
        self.training = training

        def pad_pil_image(img):
            tail = img.width % image_len_divisible_by
            if tail == 0:
                return img
            # args for pad: left, top, right, bottom
            return functional.pad(img, (0, 0, image_len_divisible_by - tail, 0), padding_mode="edge")

        def random_stretch_image(img):
            return functional.resize(img, (img.height, int(img.width * random.uniform(0.95, 1.05))))

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=4, fill=255),
            transforms.Lambda(random_stretch_image),
            transforms.Lambda(pad_pil_image),
            transforms.ToTensor()
        ])
        self.eval_transforms = transforms.ToTensor()
        self.image_len_divisible_by = image_len_divisible_by
        log = logging.getLogger(__name__)

        for uttid in tqdm(sorted(uttids)):
            imagepath = self.image_dir / f"{uttid}.jpg"
            textpath = self.trans_dir / f"{uttid}.txt"
            with open(textpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                text = " ".join(text.split())  # remove duplicated spaces
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
                right_add = np.full([img.shape[0], image_len_divisible_by - width % image_len_divisible_by, 3], 255,
                                    dtype=img.dtype)
                img = np.concatenate((img, right_add), axis=1)

            self.images.append(img)  # HxWxC -> CxHxW later
        # self.idx2idx = list(range(len(self.images)))
        if sort:
            self.sort_by_len()
            self._is_sorted = True
        self._indices = list(range(len(self.keys)))

    def __getitem__(self, index: int) -> OcrDataItem:
        index = self._indices[index]
        img = self.images[index]
        if self.training:
            img = self.train_transforms(img)
        else:
            img = self.eval_transforms(img)
        return OcrDataItem(self.keys[index], img, self.texts[index], self.encoded_texts[index],
                           self.encoded_texts[index].shape[-1])

    def __len__(self):
        return len(self.texts)

    def sort_by_len(self):
        indices = list(range(len(self.images)))
        indices.sort(key=lambda i: self.images[i].shape[1])
        self.images = [self.images[i] for i in indices]
        self.texts = [self.texts[i] for i in indices]
        self.encoded_texts = [self.encoded_texts[i] for i in indices]
        self.keys = [self.keys[i] for i in indices]

    def shuffle_buckets(self, batch_size: int, shuffle_parts=True):
        if not self._is_sorted:
            raise Exception("not sorted")
        indices = list(range(len(self.keys)))
        random.shuffle(indices)
        for i in range(0, len(indices), 100 * batch_size):
            j = min(i + 100 * batch_size, len(indices))
            indices[i:j] = sorted(indices[i:j], key=lambda idx: self.images[idx].shape[1])
        buckets = []
        for i in range(0, len(indices), batch_size):
            j = min(i + batch_size, len(indices))
            buckets.append(list(range(i, j)))
        if shuffle_parts:
            random.shuffle(buckets)
        self._indices = []
        for bucket in buckets:
            for i in bucket:
                self._indices.append(i)

    def shuffle(self):
        random.shuffle(self._indices)


def collate_fn(items: List[OcrDataItem]):
    items.sort(key=lambda item: -item.img.shape[-1])  # sort by image width
    image_lengths = torch.LongTensor([item.img.shape[-1] for item in items])
    images = utils_rnn.pad_sequence(
        [item.img.transpose(0, -1) for item in items],
        batch_first=False).permute(1, 3, 2, 0)  # CxHxW -> WxHxC -> WxBxHxC -> BxCxHxW
    encoded_texts = utils_rnn.pad_sequence([item.encoded_text for item in items], batch_first=True)
    texts = [item.text for item in items]
    keys = [item.key for item in items]
    text_lengths = torch.LongTensor([item.encoded_text_len for item in items])
    return OcrDataBatch(keys, texts, images, encoded_texts, image_lengths, text_lengths)
