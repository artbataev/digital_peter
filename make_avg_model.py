import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from digital_peter import models
from digital_peter.data import DigitalPeterDataset, collate_fn
from digital_peter.models.utils import make_avg_model_from_checkpoints, update_bn_stats
from digital_peter.text import get_chars_from_file, TextEncoder

DATA_DIR = Path(__file__).parent / "data"

if __name__ == "__main__":
    chars = get_chars_from_file(DATA_DIR / "chars_new.txt")
    encoder = TextEncoder(chars)
    num_outputs = len(encoder.id2char)
    with open(DATA_DIR / "train_uttids_set.pkl", "rb") as f:
        train_uttids = pickle.load(f)
    train_data = DigitalPeterDataset(DATA_DIR / "train", train_uttids,
                                     encoder,
                                     img_height=128, image_len_divisible_by=4,
                                     verbose=False, training=False)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, collate_fn=collate_fn)
    # exp_dir = "exp/conv__gru_2x368_drop02--h128--c62/ep-0to32_lr-0.01-1e-06-4_bs-10_optim-sgd-wd0.01"
    exp_dir = "exp/retrain/conv__gru_2x368_drop02--h128--c62/ep-0to6_lr-1e-06-1e-08-0_bs-10_optim-sgd-wd0.01"
    ep_start = 4
    ep_end = 6
    model_weigths = make_avg_model_from_checkpoints([
        f"{exp_dir}/model_ep{epoch}.pt"
        for epoch in range(ep_start, ep_end + 1)
    ])
    model = models.conv__gru_2x368_drop02(num_outputs)
    model.load_state_dict(model_weigths)
    model.cuda()
    update_bn_stats(model, train_loader)
    torch.save(model.state_dict(), f"{exp_dir}/model_ep{ep_start}-{ep_end}.pt")
