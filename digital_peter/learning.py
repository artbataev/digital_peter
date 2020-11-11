import logging

import editdistance
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class OcrLearner:
    def __init__(self, model,
                 optimizer,
                 criterion,
                 train_loader,
                 val_loader,
                 encoder,
                 logits_len_fn=lambda x: x // 4 - 1,
                 parl_decoder=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.parl_decoder = parl_decoder
        self.logits_len_fn = logits_len_fn
        self.log = logging.getLogger(__name__)

    def train_model(self):
        self.model.train()
        tmp_loss = 0.0
        tmp_loss_num = 0
        for batch_idx, (images, _, encoded_texts, image_lengths, text_lengths) in enumerate(tqdm(self.train_loader)):
            images, encoded_texts, image_lengths, text_lengths = images.cuda(), encoded_texts.cuda(), \
                                                                 image_lengths.cuda(), text_lengths.cuda()
            encoded_texts, text_lengths = encoded_texts.to(torch.int32), text_lengths.to(torch.int32)
            self.optimizer.zero_grad()
            logits = self.model(images, image_lengths)
            log_logits = F.log_softmax(logits, dim=-1)
            loss = self.criterion(log_logits, encoded_texts, self.logits_len_fn(image_lengths), text_lengths).mean()
            loss.backward()
            self.optimizer.step()

            tmp_loss_num += 1
            tmp_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                logging.info(f"loss: {tmp_loss / tmp_loss_num:.5f}")
                tmp_loss_num = 0
                tmp_loss = 0.0

    def val_model(self, greedy=True):
        if not greedy and self.parl_decoder is None:
            raise Exception("no decoder")

        self.model.eval()
        num_items = 0
        loss_accum = 0.0
        total_chars = 0
        error_chars = 0
        total_words = 0
        error_words = 0
        with torch.no_grad():
            for batch_idx, (images, texts, encoded_texts, image_lengths, text_lengths) in enumerate(
                    tqdm(self.val_loader)):
                images, encoded_texts, image_lengths, text_lengths = images.cuda(), encoded_texts.cuda(), \
                                                                     image_lengths.cuda(), text_lengths.cuda()
                encoded_texts, text_lengths = encoded_texts.to(torch.int32), text_lengths.to(torch.int32)
                logits = self.model(images, image_lengths)
                log_logits = F.log_softmax(logits, dim=-1)
                loss = self.criterion(log_logits, encoded_texts, self.logits_len_fn(image_lengths), text_lengths)
                loss_accum += loss.sum().item()
                num_items += len(texts)

                if greedy:
                    labels = logits.argmax(dim=-1).detach().cpu().numpy().transpose(1, 0)
                else:
                    beam_results, beam_scores, timesteps, out_lens = self.parl_decoder.decode(
                        log_logits.transpose(0, 1).detach(),
                        seq_lens=self.logits_len_fn(image_lengths))
                for i, ref in enumerate(texts):
                    total_chars += len(ref)
                    if greedy:
                        hyp_len = self.logits_len_fn(image_lengths[i].item())
                        hyp = self.encoder.decode_ctc(labels[i].tolist()[:hyp_len])
                    else:
                        hyp_len = out_lens[i][0]
                        hyp_encoded = beam_results[i, 0, :hyp_len]
                        hyp = self.encoder.decode(hyp_encoded.numpy().tolist())
                    error_chars += editdistance.eval(hyp, ref)
                    total_words += len(ref.split())
                    error_words += editdistance.eval(hyp.split(), ref.split())
                    if batch_idx == 0:
                        self.log.info(f"ref: {ref}")
                        self.log.info(f"hyp: {hyp}")

        loss_accum /= num_items
        self.log.info(f"loss: {loss_accum:.5f}")
        self.log.info(f"CER: {error_chars / total_chars * 100:.2f}%, WER: {error_words / total_words * 100:.2f}%")
        return loss_accum
