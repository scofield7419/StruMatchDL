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

from module import NLGTreeTrm, TopDownParser, RoIFilter
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_nlg, collate_fn_nl, collate_fn_sf, build_optimizer, get_device
from logger import Logger
from data_engine import DataEngineSplit
from tree_utils import *

from tqdm import tqdm



class StruMatchDualL:
    def __init__(
            self,
            batch_size,
            optimizer,
            learning_rate,
            train_data_engine,
            test_data_engine,
            lmbda1,lmbda2,lmbda3,
            threshold_omega,threshold_sigma,
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

        self.textA2B = NLGTreeTrm(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                attr_vocab_size=attr_vocab_size,
                vocab_size=vocab_size,
                n_layers=n_layers,
                bidirectional=bidirectional)

        self.textB2A = NLGTreeTrm(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                attr_vocab_size=attr_vocab_size,
                vocab_size=vocab_size,
                n_layers=n_layers,
                bidirectional=False)

        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
        self.lmbda3 = lmbda3

        self.threshold_omega = threshold_omega
        self.threshold_sigma = threshold_sigma

        self.textA2B.to(self.device)
        self.textB2A.to(self.device)

        self.RoI_filter = RoIFilter(dim_hidden,dim_hidden)
        tag_vocab = Vocabulary()
        tag_vocab.index(START)
        tag_vocab.index(STOP)

        word_vocab = Vocabulary()
        word_vocab.index(START)
        word_vocab.index(STOP)
        word_vocab.index(UNK)

        label_vocab = Vocabulary()
        label_vocab.index(())

        self.rec_parserA = TopDownParser(
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim=50,
            word_embedding_dim=dim_embedding,
            lstm_layers=1,
            lstm_dim=dim_hidden,
            label_hidden_dim=dim_hidden,
            split_hidden_dim=dim_hidden,
            dropout=0.5,
        )

        self.rec_parserB = TopDownParser(
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim=50,
            word_embedding_dim=dim_embedding,
            lstm_layers=1,
            lstm_dim=dim_hidden,
            label_hidden_dim=dim_hidden,
            split_hidden_dim=dim_hidden,
            dropout=0.5,
        )

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

    def train(self, epochs, batch_size, criterion, RoI_align_loss,
              save_epochs=10,
              teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              max_norm=0.25):

        self.batches = 0

        for idx in range(1, epochs+1):
            epoch_text2text_loss = epoch_dual_loss = \
                epoch_struc_loss = epoch_rec_loss = 0
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
                textB2A_logits, textB2A_outputs, textB2A_targets, textB2A_RoIs = self.run_nlg_batch(
                        self.textB2A, batch,
                        scorer=textB2A_scorer,
                        testing=False,
                        teacher_forcing_ratio=teacher_forcing_ratio
                )

                textA2B_logits, textA2B_outputs, textA2B_targets, textA2B_RoIs = self.run_nlg_batch(
                        self.textA2B,batch,
                        scorer=textA2B_scorer,
                        testing=False,
                        teacher_forcing_ratio=teacher_forcing_ratio
                )

                textA2B_RoIs, textB2A_RoIs = self.RoI_aligning(textA2B_RoIs, textB2A_RoIs,
                                                               self.threshold_omega, self.threshold_sigma)
                struc_loss = RoI_align_loss(textA2B_RoIs, textB2A_RoIs)
                gold_treesA = []
                gold_treesB = []
                sentences_A = []
                sentences_B = []

                for batch_ in batch:
                    _, tree_A, _, tree_B, _, _ = batch_
                    gold_treesA.append(tree_A)
                    gold_treesB.append(tree_B)
                    sentence_A = [(leaf.tag, leaf.word) for leaf in tree_A.leaves()]
                    sentences_A.append(sentence_A)
                    sentence_B = [(leaf.tag, leaf.word) for leaf in tree_B.leaves()]
                    sentences_B.append(sentence_B)
                _, rec_batch_losses_A = self.rec_parserA.parse(sentences_A, gold_treesA)
                _, rec_batch_losses_B = self.rec_parserA.parse(sentences_B, gold_treesB)
                rec_loss = rec_batch_losses_A + rec_batch_losses_B

                text2text_loss, dual_loss = criterion(
                        textB2A_logits.cpu(),
                        textB2A_outputs.cpu(),
                        textA2B_logits.cpu(),
                        textB2A_targets.cpu(),
                        textA2B_targets.cpu()
                )

                total_batch_loss = text2text_loss + self.lmbda1 * dual_loss + \
                            self.lmbda2 * struc_loss + self.lmbda3 * rec_loss

                total_batch_loss.backward(retain_graph=True)
                # dual_loss.backward(retain_graph=True)
                # struc_loss.backward(retain_graph=True)
                self.textA2B_optimizer.step()
                self.textB2A_optimizer.step()
                self.textA2B_optimizer.zero_grad()
                self.textB2A_optimizer.zero_grad()

                batch_amount += 1
                epoch_text2text_loss += text2text_loss.item()
                epoch_dual_loss += dual_loss.item()
                epoch_struc_loss += struc_loss.item()
                epoch_rec_loss += rec_loss.item()
                pbar.set_postfix(
                        TextLoss="{:.4f}".format(epoch_text2text_loss / batch_amount),
                        DualLoss="{:.4f}".format(epoch_dual_loss / batch_amount),
                        MatchLoss="{:.4f}".format(epoch_struc_loss / batch_amount),
                        ReconLoss="{:.4f}".format(epoch_rec_loss / batch_amount),
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

    def RoI_aligning(self, textA2B_RoIs, textB2A_RoIs,
                     threshold_omega=0.30, threshold_sigma=0.50):
        textA2B_RoIs = self.RoI_filter(textA2B_RoIs, threshold_omega)
        textB2A_RoIs = self.RoI_filter(textB2A_RoIs, threshold_omega)

        good_pairs = []
        index_A = []
        index_B = []
        for a_i, a_r in enumerate(textA2B_RoIs):
            for b_i, b_r in enumerate(textB2A_RoIs):
                sim = cosine_similarity(a_r, b_r)
                if sim >= threshold_sigma:
                    good_pairs.append((a_i, b_i))
                    index_A.append(a_i)
                    index_B.append(b_i)
        index_A = torch.LongTensor(index_A)
        index_B = torch.LongTensor(index_B)
        return torch.gather(textA2B_RoIs, dim=1, index=index_A), \
               torch.gather(textB2A_RoIs, dim=1, index=index_B)


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
                textA2B_logits, textA2B_outputs, textA2B_targets, textA2B_RoIs = self.run_nlg_batch(
                        self.textA2B, batch,
                        scorer=textA2B_scorer,
                        testing=True
                )
                textB2A_logits, textB2A_outputs, textB2A_targets, textB2A_RoIs = self.run_nlg_batch(
                        self.textB2A, batch,
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

    def run_nlg_batch(self, model, batch, scorer=None,
                      testing=False, teacher_forcing_ratio=0.5,
                      result_path=None):
        if testing:
            model.eval()
        else:
            model.train()

        encoder_input, _, decoder_label, _, refs, sf_data = batch

        attrs = self._sequences_to_nhot(encoder_input, self.attr_vocab_size)
        attrs = torch.from_numpy(attrs).to(self.device)
        labels = torch.from_numpy(decoder_label).to(self.device)

        # logits.size() == (batch_size, 1, seq_length, vocab_size)
        # outputs.size() == (batch_size, 1, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph
        logits, outputs, _RoIs = model(
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
        return logits, outputs, labels, _RoIs

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
