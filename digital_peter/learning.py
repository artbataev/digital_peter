import logging
from collections import defaultdict
from typing import Dict

import editdistance
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from digital_peter.data import OcrDataBatch
from digital_peter.text import calc_metrics


class OcrLearner:
    def __init__(self, model,
                 optimizer,
                 criterion,
                 train_loader,
                 val_loader,
                 encoder,
                 parl_decoder=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.parl_decoder = parl_decoder
        self.log = logging.getLogger(__name__)

    def train_model(self, reduce_lr=None):
        self.model.train()
        tmp_loss = 0.0
        tmp_loss_num = 0
        ocr_data_batch: OcrDataBatch
        for batch_idx, ocr_data_batch in enumerate(tqdm(self.train_loader)):
            images = ocr_data_batch.images.cuda()
            image_lengths = ocr_data_batch.image_lengths.cuda()
            encoded_texts = ocr_data_batch.encoded_texts.cuda().to(torch.int32)  # for ctc
            text_lengths = ocr_data_batch.text_lengths.cuda().to(torch.int32)  # for ctc

            self.optimizer.zero_grad()
            logits, logits_lengths = self.model(images, image_lengths)
            log_logits = F.log_softmax(logits, dim=-1)
            loss = self.criterion(log_logits, encoded_texts, logits_lengths, text_lengths).mean()
            loss.backward()
            self.optimizer.step()

            tmp_loss_num += 1
            tmp_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                logging.info(f"loss: {tmp_loss / tmp_loss_num:.5f}")
                tmp_loss_num = 0
                tmp_loss = 0.0
            if reduce_lr is not None:
                reduce_lr.step()

    def get_val_logits(self):
        self.model.eval()
        utt2log_logits: Dict[str, torch.Tensor] = dict()
        with torch.no_grad():
            ocr_data_batch: OcrDataBatch
            for batch_idx, ocr_data_batch in enumerate(tqdm(self.val_loader)):
                images = ocr_data_batch.images.cuda()
                image_lengths = ocr_data_batch.image_lengths.cuda()
                batch_size = images.shape[0]

                logits, logits_lengths = self.model(images, image_lengths)
                log_logits = F.log_softmax(logits, dim=-1)

                # save logits
                cpu_log_logits = log_logits.transpose(0, 1).detach().cpu()
                for i in range(batch_size):
                    key = ocr_data_batch.keys[i]
                    cur_logits = cpu_log_logits[i, :logits_lengths[i]].detach()
                    utt2log_logits[key] = cur_logits
        return utt2log_logits

    def val_model(self, greedy=True):
        if not greedy and self.parl_decoder is None:
            raise Exception("no decoder")

        self.model.eval()
        num_items = 0
        loss_accum = 0.0

        utt2log_logits: Dict[str, torch.Tensor] = dict()
        utt2hyp: Dict[str, str] = dict()
        utt2ref: Dict[str, str] = dict()
        with torch.no_grad():
            ocr_data_batch: OcrDataBatch
            for batch_idx, ocr_data_batch in enumerate(tqdm(self.val_loader)):
                images = ocr_data_batch.images.cuda()
                image_lengths = ocr_data_batch.image_lengths.cuda()
                encoded_texts = ocr_data_batch.encoded_texts.cuda().to(torch.int32)  # for ctc
                text_lengths = ocr_data_batch.text_lengths.cuda().to(torch.int32)  # for ctc
                batch_size = images.shape[0]

                logits, logits_lengths = self.model(images, image_lengths)
                log_logits = F.log_softmax(logits, dim=-1)
                loss = self.criterion(log_logits, encoded_texts, logits_lengths, text_lengths)
                loss_accum += loss.sum().item()
                num_items += batch_size

                # save logits
                cpu_log_logits = log_logits.transpose(0, 1).detach().cpu()
                for i in range(batch_size):
                    key = ocr_data_batch.keys[i]
                    cur_logits = cpu_log_logits[i, :logits_lengths[i]].detach()
                    utt2log_logits[key] = cur_logits
                    utt2ref[key] = ocr_data_batch.texts[i]

                if greedy:
                    labels = logits.argmax(dim=-1).detach().cpu().numpy().transpose(1, 0)
                else:
                    beam_results, beam_scores, timesteps, out_lens = self.parl_decoder.decode(
                        log_logits.transpose(0, 1).detach(),
                        seq_lens=logits_lengths)

                for i, ref in enumerate(ocr_data_batch.texts):
                    key = ocr_data_batch.keys[i]
                    if greedy:
                        hyp_len = logits_lengths[i].item()
                        hyp = self.encoder.decode_ctc(labels[i].tolist()[:hyp_len]).strip()
                    else:
                        hyp_len = out_lens[i][0]
                        hyp_encoded = beam_results[i, 0, :hyp_len]
                        hyp = self.encoder.decode(hyp_encoded.numpy().tolist()).strip()
                    utt2hyp[key] = hyp

        loss_accum /= num_items
        self.log.info(f"loss: {loss_accum:.5f}")
        _ = calc_metrics(utt2hyp, utt2ref)
        return loss_accum

        # not used now

        # re-decode
        utt2hyp_old = utt2hyp
        utt2hyp = dict()

        merged_keys = defaultdict(list)
        for key in utt2ref:
            key_base, line = key.rsplit("_", maxsplit=1)
            merged_keys[key_base].append(int(line))

        for mkey, lines in tqdm(merged_keys.items()):
            keys = [f"{mkey}_{line}" for line in sorted(lines)]
            log_logits = [utt2log_logits[key] for key in keys]
            cum_logits_lengths = [l.shape[0] for l in log_logits]
            for i in range(1, len(cum_logits_lengths)):
                cum_logits_lengths[i] += cum_logits_lengths[i - 1]
            stacked_logits = torch.cat(log_logits, 0).unsqueeze(0)
            beam_results, beam_scores, timesteps, out_lens = self.parl_decoder.decode(
                stacked_logits, seq_lens=torch.LongTensor([stacked_logits.shape[1]]))
            l = out_lens[0][0]
            result = beam_results[0][0][:l]
            timesteps = timesteps[0][0][:l]
            utterances = [[] for _ in range(len(keys))]
            cur = 0
            for i in range(l):
                if timesteps[i] >= cum_logits_lengths[cur]:
                    cur += 1
                utterances[cur].append(result[i].item())
            for key, raw_hyp in zip(keys, utterances):
                utt2hyp[key] = self.encoder.decode(raw_hyp).strip()

        _ = calc_metrics(utt2hyp, utt2ref)
        for key in sorted(utt2hyp.keys()):
            hyp = utt2hyp[key]
            hyp_old = utt2hyp_old[key]
            ref = utt2ref[key]
            print(f"{key}: {hyp_old} -> {hyp} | {ref}")
            print(f"{editdistance.eval(hyp_old, ref)} -> {editdistance.eval(hyp, ref)}")
        return loss_accum
