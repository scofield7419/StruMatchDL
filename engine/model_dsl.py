import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import random
import numpy as np
import os
import math

from module import NLGTrm
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_nlg, build_optimizer, get_device
from logger import Logger

from tqdm import tqdm


class DSL:
    def __init__(
            self,
            batch_size,
            optimizer,
            learning_rate,
            train_data_engine,
            test_data_engine,
            dim_hidden,
            dim_embedding,
            vocab_size=None,
            attr_vocab_size=None,
            n_layers=12,
            bidirectional=False,
            model_dir="./model",
            log_dir="./log",
            is_load=True,
            replace_model=True,
            device=None,
            dir_name='test'
    ):

        # Initialize attributes
        self.data_engine = train_data_engine
        self.n_layers = n_layers
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.dim_hidden = dim_hidden
        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size
        self.attr_vocab_size = attr_vocab_size
        self.dir_name = dir_name

        self.device = get_device(device)

        self.textA2B = NLGTrm(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                attr_vocab_size=attr_vocab_size,
                vocab_size=vocab_size,
                n_layers=n_layers,
                bidirectional=bidirectional)

        self.textB2A = NLGTrm(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                attr_vocab_size=attr_vocab_size,
                vocab_size=vocab_size,
                n_layers=n_layers,
                bidirectional=False)

        self.textA2B.to(self.device)
        self.textB2A.to(self.device)

        # Initialize data loaders and optimizers
        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine
        self.train_data_loader = DataLoader(
                train_data_engine,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn_nlg,
                pin_memory=True)

        self.test_data_loader = DataLoader(
                test_data_engine,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn_nlg,
                pin_memory=True)

        self.textA2B_parameters = filter(
                lambda p: p.requires_grad, self.textA2B.parameters())
        self.textA2B_optimizer = build_optimizer(
                optimizer, self.textA2B_parameters,
                learning_rate)

        self.textB2A_parameters = filter(
                lambda p: p.requires_grad, self.textB2A.parameters())
        self.textB2A_optimizer = build_optimizer(
                optimizer, self.textB2A_parameters,
                learning_rate)

        print_time_info("Model create complete")

        self.model_dir, self.log_dir = handle_model_dirs(
            model_dir, log_dir, dir_name, replace_model, is_load
        )

        if is_load:
            self.load_model(self.model_dir)

        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_log_path = os.path.join(
                self.log_dir, "valid_log.csv")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch,textA2B_loss,textB2A_loss,micro_f1,"
                       "bleu,rouge(1,2,L,BE)\n")
        with open(self.valid_log_path, 'w') as file:
            file.write("epoch,textA2B_loss,textB2A_loss,micro_f1, "
                       "bleu,rouge(1,2,L,BE)\n")

        # Initialize batch count
        self.batches = 0

    def train(self, epochs, batch_size, criterion,
              save_epochs=10,
              teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              max_norm=0.25):

        self.batches = 0

        for idx in range(1, epochs+1):
            epoch_textB2A_loss = epoch_textA2B_loss = epoch_dual_loss = 0
            batch_amount = 0
            textA2B_scorer = SequenceScorer()
            textB2A_scorer = SequenceScorer()

            pbar = tqdm(
                self.train_data_loader,
                total=len(self.train_data_loader),
                dynamic_ncols=True
            )

            for batch in pbar:
                self.batches += 1
                textB2A_logits, textB2A_outputs, textB2A_targets = self.run_nlg_batch(
                        batch,
                        scorer=textB2A_scorer,
                        testing=False,
                        teacher_forcing_ratio=teacher_forcing_ratio
                )

                textA2B_logits, textA2B_outputs, textA2B_targets = self.run_nlg_batch(
                        batch,
                        scorer=textA2B_scorer,
                        testing=False,
                        teacher_forcing_ratio=teacher_forcing_ratio
                )

                textB2A_loss, textA2B_loss, dual_loss = criterion(
                        textB2A_logits.cpu(),
                        textB2A_outputs.cpu(),
                        textA2B_logits.cpu(),
                        textB2A_targets.cpu(),
                        textA2B_targets.cpu()
                )

                textB2A_loss.backward(retain_graph=True)
                textA2B_loss.backward(retain_graph=False)
                self.textA2B_optimizer.step()
                self.textB2A_optimizer.step()
                self.textA2B_optimizer.zero_grad()
                self.textB2A_optimizer.zero_grad()

                batch_amount += 1
                epoch_textA2B_loss += textA2B_loss.item()
                epoch_textB2A_loss += textB2A_loss.item()
                epoch_dual_loss += dual_loss.item()
                pbar.set_postfix(
                        ULoss="{:.4f}".format(epoch_textA2B_loss / batch_amount),
                        GLoss="{:.3f}".format(epoch_textB2A_loss / batch_amount),
                        DLoss="{:.4f}".format(epoch_dual_loss / batch_amount),
                )

            textB2A_scorer.print_avg_scores()
            textA2B_scorer.print_avg_scores()

            # save model
            if idx % save_epochs == 0:
                print_time_info(f"Epoch {idx}: save model...")
                self.save_model(self.model_dir)

            self._record_log(
                epoch=idx,
                testing=False,
                textA2B_loss=epoch_textA2B_loss,
                textB2A_loss=epoch_textB2A_loss,
                textA2B_scorer=textA2B_scorer,
                textB2A_scorer=textB2A_scorer
            )

            self.test(
                batch_size=batch_size,
                criterion=criterion,
                epoch=idx
            )

            teacher_forcing_ratio *= tf_decay_rate
            criterion.epoch_end()

        return (
            textB2A_loss / batch_amount,
            textA2B_loss / batch_amount,
            textB2A_scorer,
            textA2B_scorer,
        )

    def test(self, batch_size,
             criterion, epoch=-1):

        batch_amount = 0

        textA2B_scorer = SequenceScorer()
        textA2B_loss = 0
        textB2A_scorer = SequenceScorer()
        textB2A_loss = 0
        dual_loss = 0
        for b_idx, batch in enumerate(tqdm(self.test_data_loader)):
            with torch.no_grad():
                textA2B_logits, textA2B_outputs, textA2B_targets = self.run_nlg_batch(
                        batch,
                        scorer=textA2B_scorer,
                        testing=True
                )
                textB2A_logits, textB2A_outputs, textB2A_targets = self.run_nlg_batch(
                        batch,
                        scorer=textB2A_scorer,
                        testing=True,
                        teacher_forcing_ratio=0.0,
                        result_path=os.path.join(
                            os.path.join(self.log_dir, "validation"),
                            "test.txt"
                        )
                )
                batch_textB2A_loss, batch_textA2B_loss, batch_dual_loss = criterion(
                        textB2A_logits.cpu(),
                        textB2A_outputs.cpu(),
                        textA2B_logits.cpu(),
                        textB2A_targets.cpu(),
                        textA2B_targets.cpu()
                )

            textA2B_loss += batch_textA2B_loss.item()
            textB2A_loss += batch_textB2A_loss.item()
            dual_loss += batch_dual_loss.item()
            batch_amount += 1

        textA2B_loss /= batch_amount
        textA2B_scorer.print_avg_scores()
        textB2A_loss /= batch_amount
        textB2A_scorer.print_avg_scores()
        dual_loss /= batch_amount

        self._record_log(
            epoch=epoch,
            testing=True,
            textA2B_loss=textA2B_loss,
            textB2A_loss=textB2A_loss,
            dual_loss=dual_loss,
            textA2B_scorer=textA2B_scorer,
            textB2A_scorer=textB2A_scorer
        )

        with open("test_results.txt", 'a') as file:
            file.write("{}\n".format(self.dir_name))
            textB2A_scorer.write_avg_scores_to_file(file)
            textA2B_scorer.write_avg_scores_to_file(file)

    def run_nlg_batch(self, batch, scorer=None,
                      testing=False, teacher_forcing_ratio=0.5,
                      result_path=None):
        if testing:
            self.nlg.eval()
        else:
            self.nlg.train()

        encoder_input, decoder_label, refs, sf_data = batch

        attrs = self._sequences_to_nhot(encoder_input, self.attr_vocab_size)
        attrs = torch.from_numpy(attrs).to(self.device)
        labels = torch.from_numpy(decoder_label).to(self.device)

        # logits.size() == (batch_size, 1, seq_length, vocab_size)
        # outputs.size() == (batch_size, 1, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph
        logits, outputs = self.nlg(
            attrs, _BOS, labels, beam_size=1,
            tf_ratio=teacher_forcing_ratio if not testing else 0.0
        )

        logits = logits.squeeze(1)
        outputs = outputs.squeeze(1)
        batch_size, seq_length, vocab_size = logits.size()

        outputs_indices = outputs.detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        if scorer:
            labels_clone = labels.detach().cpu().numpy()
            scorer.update(labels_clone, refs, outputs_indices)

        if testing and result_path:
            self._record_nlg_test_result(
                result_path,
                encoder_input,
                decoder_label,
                sf_data,
                outputs_indices
            )
        return logits, outputs, labels

    def save_model(self, model_dir):
        textA2B_path = os.path.join(model_dir, "textA2B.ckpt")
        textB2A_path = os.path.join(model_dir, "textB2A.ckpt")
        torch.save(self.textA2B, textA2B_path)
        torch.save(self.textB2A, textB2A_path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        # Get the latest modified model (files or directory)
        textA2B_path = os.path.join(model_dir, "textA2B.ckpt")
        textB2A_path = os.path.join(model_dir, "textB2A.ckpt")

        if not os.path.exists(textA2B_path) or not os.path.exists(textB2A_path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.textA2B = torch.load(textA2B_path, map_location=self.device)
            self.textB2A = torch.load(textB2A_path, map_location=self.device)
            print_time_info(f"Load model from {model_dir} successfully")

    def _sequences_to_nhot(self, seqs, vocab_size):
        """
        args:
            seqs: list of list of word_ids
            vocab_size: int

        outputs:
            labels: np.array of shape [batch_size, vocab_size]
        """
        labels = np.zeros((len(seqs), vocab_size), dtype=np.int)
        for bid, seq in enumerate(seqs):
            for word in seq:
                labels[bid][word] = 1
        return labels

    def _record_log(self,
                    epoch,
                    testing,
                    textA2B_loss=None,
                    textB2A_loss=None,
                    dual_loss=None,
                    textA2B_scorer=None,
                    textB2A_scorer=None):
        filename = self.valid_log_path if testing else self.train_log_path
        textA2B_loss = 'None' if textA2B_loss is None else '{:.4f}'.format(textA2B_loss)
        textB2A_loss = 'None' if textB2A_loss is None else '{:.3f}'.format(textB2A_loss)
        dual_loss = 'None' if dual_loss is None else '{:.4f}'.format(dual_loss)
        if textA2B_scorer is not None:
            micro_f1, _ = textA2B_scorer.get_avg_scores()
            micro_f1 = '{:.4f}'.format(micro_f1)
        else:
            micro_f1 = '-1.0'
        if textB2A_scorer is not None:
            _, bleu, _, rouge, _ = textB2A_scorer.get_avg_scores()
            bleu = '{:.4f}'.format(bleu)
            rouge = ' '.join(['{:.4f}'.format(s) for s in rouge])
        else:
            bleu, rouge = '-1.0', '-1.0 -1.0 -1.0'
        with open(filename, 'a') as file:
            file.write(f"{epoch},{textA2B_loss},{textB2A_loss},{micro_f1},"
                       f"{bleu},{rouge},{dual_loss}\n")

    def _record_textA2B_test_result(self,
                                result_path,
                                encoder_input,
                                decoder_label,
                                prediction):
        '''
        encoder_input = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx], is_token=True)
            for idx, sent in enumerate(encoder_input)
        ]
        decoder_label = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_label)
        ]
        decoder_result = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_result)
        ]

        with open(result_path, 'a') as file:
            for idx in range(len(encoder_input)):
                file.write("---------\n")
                file.write(f"Data {idx}\n")
                file.write(f"encoder input: {' '.join(encoder_input[idx])}\n")
                file.write(f"decoder output: {' '.join(decoder_result[idx])}\n")
                file.write(f"decoder label: {' '.join(decoder_label[idx])}\n")
        '''
        pass

    def _record_nlg_test_result(self, result_path, encoder_input,
                            decoder_label, sf_data, decoder_result):
        encoder_input = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx], is_token=True)
            for idx, sent in enumerate(encoder_input)
        ]
        decoder_label = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_label)
        ]
        decoder_result = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_result)
        ]

        with open(result_path, 'a') as file:
            for idx in range(len(encoder_input)):
                file.write("---------\n")
                file.write(f"Data {idx}\n")
                file.write(f"encoder input: {' '.join(encoder_input[idx])}\n")
                file.write(f"decoder output: {' '.join(decoder_result[idx])}\n")
                file.write(f"decoder label: {' '.join(decoder_label[idx])}\n")

