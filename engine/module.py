import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyemd import emd_samples
from sklearn.metrics import f1_score
from utils import *

import math
import torch
import torch.nn as nn
from torch.autograd import Variable as Var
from torch.nn import Parameter
from Engines.TreeUtils import *
import numpy as np
from tree_utils import *


from torch.autograd import Variable
from torch.nn import init

class Criterion(nn.Module):
    def __init__(self, model, reward_type, loss_weight,
                 supervised=True, rl_lambda=1.0, rl_alpha=0.5,
                 pretrain_epochs=0, total_epochs=-1, anneal_type='none',
                 LM=None, LM2=None, training_set_label_samples=None):
        super(Criterion, self).__init__()
        self.model = model
        self.reward_type = reward_type
        self.supervised = supervised
        self.rl_lambda = rl_lambda
        self.rl_alpha = rl_alpha
        self.pretrain_epochs = pretrain_epochs
        self.epoch = 0
        self.total_epochs = total_epochs
        self.anneal_type = anneal_type
        if anneal_type == 'linear' and (total_epochs is None):
            raise ValueError("Please set total_epochs if you want to " \
                             "use anneal_type='linear'")
        if anneal_type == 'switch' and pretrain_epochs == 0:
            raise ValueError("Please set pretrain_epochs > 0 if you want to " \
                             "use anneal_type='switch'")
        self.LM = LM
        self.LM2 = LM2
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.CE = nn.CrossEntropyLoss(weight=loss_weight, reduction='none')
        if 'em' in reward_type:
            samples = sum(training_set_label_samples, [])
            np.random.shuffle(samples)
            n = 10
            size = len(samples) // n
            self.samples = [
                samples[i*size:(i+1)*size]
                for i in range(n)
            ]

    def set_scorer(self, scorer):
        self.scorer = scorer

    def epoch_end(self):
        self.epoch += 1
        if self.anneal_type != 'none' and self.epoch == self.pretrain_epochs:
            print_time_info("loss scheduling started ({})".format(self.anneal_type))

    def earth_mover(self, decisions):
        # decisions.size() == (batch_size, sample_size, attr_vocab_size)
        length = decisions.size(-1)
        indexes = (decisions.float().numpy() >= 0.5)
        emd = [
            [
                emd_samples(
                    np.arange(length)[index].tolist(),
                    self.samples[0]
                ) if index.sum() > 0 else 1.0
                for index in indexes[bid]
            ]
            for bid in range(decisions.size(0))
        ]
        return torch.tensor(emd, dtype=torch.float, device=decisions.device)

    def get_scheduled_loss(self, sup_loss, rl_loss):
        if self.epoch < self.pretrain_epochs:
            return sup_loss, 0
        elif self.anneal_type == 'none':
            return sup_loss, rl_loss
        elif self.anneal_type == 'switch':
            return 0, rl_loss

        assert self.anneal_type == 'linear'
        rl_weight = (self.epoch - self.pretrain_epochs + 1) / (self.total_epochs - self.pretrain_epochs + 1)
        return (1-rl_weight) * sup_loss, rl_weight * rl_loss

    def get_scores(self, name, logits):
        size = logits.size(0)
        ret = torch.tensor(getattr(self.scorer, name)[-size:]).float()
        if len(ret.size()) == 2:
            ret = ret.mean(dim=-1)
        return ret

    def get_log_joint_prob_nlg(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, beam_size, seq_length, vocab_size]
            decisions: tensor of shape [batch_size, beam_size, seq_length, vocab_size]
                       one-hot vector of decoded word-ids
        returns:
            log_joint_prob: tensor of shape [batch_size, beam_size]
        """
        logits = logits.contiguous().view(*decisions.size())
        probs = torch.softmax(logits, dim=-1)
        return (decisions * probs).sum(dim=-1).log().sum(dim=-1)

    def get_log_joint_prob_nlu(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, attr_vocab_size]
                    or [batch_size, sample_size, attr_vocab_size]
            decisions: tensor of shape [batch_size, sample_size, attr_vocab_size]
                       decisions(0/1)
        returns:
            log_joint_prob: tensor of shape [batch_size, sample_size]
        """
        if len(logits.size()) == len(decisions.size()) - 1:
            logits = logits.unsqueeze(1).expand(-1, decisions.size(1), -1)

        probs = torch.sigmoid(logits)
        decisions = decisions.float()
        probs = probs * decisions + (1-probs) * (1-decisions)
        return probs.log().sum(dim=-1)

    def lm_log_prob(self, decisions):
        # decisions.size() == (batch_size, beam_size, seq_length, vocab_size)
        log_probs = [
            self.LM.get_log_prob(decisions[:, i])
            for i in range(decisions.size(1))
        ]
        return torch.stack(log_probs, dim=0).transpose(0, 1)

    def made_log_prob(self, decisions):
        log_probs = [
            self.MADE.get_log_prob(decisions[:, i].float())
            for i in range(decisions.size(1))
        ]
        return torch.stack(log_probs, dim=0).transpose(0, 1)

    def nlg_loss(self, logits, targets):
        bs = targets.size(0)
        loss = [
            self.CE(logits[:, i].contiguous().view(-1, logits.size(-1)), targets.view(-1)).view(bs, -1).mean(-1)
            for i in range(logits.size(1))
        ]
        return torch.stack(loss, dim=0).transpose(0, 1)

    def nlg_score(self, decisions, targets, func):
        scores = [
            func(targets, np.argmax(decisions.detach().cpu().numpy()[:, i], axis=-1))
            for i in range(decisions.size(1))
        ]
        scores = torch.tensor(scores, dtype=torch.float, device=decisions.device).transpose(0, 1)
        if len(scores.size()) == 3:
            scores = scores.mean(-1)

        return scores

    def nlu_loss(self, logits, targets):
        loss = [
            self.BCE(logits[:, i], targets).mean(-1)
            for i in range(logits.size(1))
        ]
        return torch.stack(loss, dim=0).transpose(0, 1)

    def nlu_score(self, decisions, targets, average):
        device = decisions.device
        decisions = decisions.detach().cpu().long().numpy()
        targets = targets.detach().cpu().long().numpy()
        scores = [
            [
                f1_score(y_true=np.array([label]), y_pred=np.array([pred]), average=average)
                for label, pred in zip(targets, decisions[:, i])
            ]
            for i in range(decisions.shape[1])
        ]
        return torch.tensor(scores, dtype=torch.float, device=device).transpose(0, 1)

    def get_reward(self, logits, targets, decisions=None):
        reward = 0
        if decisions is not None:
            decisions = decisions.detach()

        if self.model == "nlu":
            if self.reward_type == "loss":
                reward = self.nlu_loss(logits, targets)
            elif self.reward_type == "micro-f1":
                reward = -self.nlu_score(decisions, targets, 'micro')
            elif self.reward_type == "weighted-f1":
                reward = -self.nlu_score(decisions, targets, 'weighted')
            elif self.reward_type == "f1":
                reward = -(self.nlu_score(decisions, targets, 'micro') + self.nlu_score(decisions, targets, 'weighted'))
            elif self.reward_type == "em":
                reward = self.earth_mover(decisions)
            elif self.reward_type == "made":
                reward = -self.made_log_prob(decisions)
            elif self.reward_type == "loss-em":
                reward = self.nlu_loss(logits, targets) + self.earth_mover(decisions)
        elif self.model == "nlg":
            if self.reward_type == "loss":
                reward = self.nlg_loss(logits, targets)
            elif self.reward_type == "lm":
                reward = -self.lm_log_prob(decisions)
            elif self.reward_type == "bleu":
                reward = -self.nlg_score(decisions, targets, func=single_BLEU)
            elif self.reward_type == "rouge":
                reward = -self.nlg_score(decisions, targets, func=single_ROUGE)
            elif self.reward_type == "bleu-rouge":
                reward = -(self.nlg_score(decisions, targets, func=single_BLEU) + self.nlg_score(decisions, targets, func=single_ROUGE))
            elif self.reward_type == "loss-lm":
                reward = self.nlg_loss(logits, targets) - self.lm_log_prob(decisions)

        return reward

    def forward(self, logits, targets, decisions=None, n_supervise=1,
                log_joint_prob=None, supervised=True, last_reward=0.0, calculate_reward=True):
        """
        args:
            logits: tensor of shape [batch_size, sample_size, * ]
            targets: tensor of shape [batch_size, *]
            decisions: tensor of shape [batch_size, sample_size, *]
        """
        if not self.supervised:
            supervised = False

        logits = logits.contiguous()
        targets = targets.contiguous()

        sup_loss = rl_loss = 0
        reward = 0.0
        if self.epoch >= self.pretrain_epochs and calculate_reward:
            reward = self.rl_lambda * self.get_reward(logits, targets, decisions)
        if isinstance(last_reward, torch.Tensor):
            reward = self.rl_alpha * last_reward + (1 - self.rl_alpha) * reward

        if self.model == "nlu":
            if supervised:
                splits = logits.split(split_size=1, dim=1)
                for i in range(n_supervise):
                    sup_loss += self.BCE(splits[i].squeeze(1), targets).mean()
            X = self.get_log_joint_prob_nlu(logits, decisions) if log_joint_prob is None else log_joint_prob
        elif self.model == "nlg":
            if supervised:
                splits = logits.split(split_size=1, dim=1)
                for i in range(n_supervise):
                    sup_loss += self.CE(splits[i].contiguous().view(-1, logits.size(-1)), targets.view(-1)).mean()
            X = self.get_log_joint_prob_nlg(logits, decisions) if log_joint_prob is None else log_joint_prob

        if isinstance(reward, torch.Tensor):
            rl_loss = (reward * X).mean()

        sup_loss, rl_loss = self.get_scheduled_loss(sup_loss, rl_loss)

        return sup_loss, rl_loss, X, reward


class RNNModel(nn.Module):
    def __init__(self,
                 dim_embedding,
                 dim_hidden,
                 attr_vocab_size,
                 vocab_size,
                 n_layers=1,
                 bidirectional=False):
        super(RNNModel, self).__init__()
        if attr_vocab_size and attr_vocab_size > dim_hidden:
            raise ValueError(
                "attr_vocab_size ({}) should be no larger than "
                "dim_hidden ({})".format(attr_vocab_size, dim_hidden)
            )
        self.dim_embedding = dim_embedding
        self.dim_hidden = dim_hidden
        self.attr_vocab_size = attr_vocab_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.rnn = nn.GRU(dim_embedding,
                          dim_hidden,
                          num_layers=n_layers,
                          batch_first=True,
                          bidirectional=bidirectional)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _init_hidden(self, inputs):
        """
        args:
            inputs: shape [batch_size, *]
                    a input tensor with correct device

        returns:
            hidden: shpae [n_layers*n_directions, batch_size, dim_hidden]
                    all-zero hidden state
        """
        batch_size = inputs.size(0)
        return torch.zeros(self.n_layers*self.n_directions,
                           batch_size,
                           self.dim_hidden,
                           dtype=torch.float,
                           device=inputs.device)

    def _init_hidden_with_attrs(self, attrs):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size], a n-hot vector

        returns:
            hidden: shape [n_layers*n_directions, batch_size, dim_hidden]
        """
        batch_size = attrs.size(0)
        hidden = torch.cat(
            [
                attrs,
                torch.zeros(batch_size,
                            self.dim_hidden - self.attr_vocab_size,
                            dtype=attrs.dtype,
                            device=attrs.device)
            ], 1)
        '''
        # ignore _UNK and _PAD
        hidden[:, 0:2] = 0
        '''
        return hidden.unsqueeze(0).expand(self.n_layers*self.n_directions, -1, -1).float()


class NLGRNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(NLGRNN, self).__init__(*args, **kwargs)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in NLG model.")

        self.transform = nn.Linear(self.attr_vocab_size, self.dim_hidden)
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def _st_softmax(self, logits, hard=False, dim=-1):
        y_soft = logits.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def _st_onehot(self, logits, indices, hard=True, dim=-1):
        y_soft = logits.softmax(dim)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(logits.device)
        if len(logits.size()) == len(indices.size()) + 1:
            indices = indices.unsqueeze(-1)
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        if hard:
            return y_hard - y_soft.detach() + y_soft, y_hard
        else:
            return y_soft, y_hard

    def forward(self, attrs, bos_id, labels=None,
                tf_ratio=0.5, max_decode_length=50, beam_size=5, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, beam_size, seq_length, vocab_size]
            outputs: shape [batch_size, beam_size, seq_length, vocab_size]
                     output words as one-hot vectors (maybe soft)
            decisions: shape [batch_size, beam_size, seq_length, vocab_size]
                       output words as one-hot vectors (hard)
        """
        if beam_size == 1:
            logits, outputs = self.forward_greedy(
                attrs, bos_id, labels,
                tf_ratio=tf_ratio, max_decode_length=max_decode_length,
                st=st
            )
            return logits.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1)

        decode_length = max_decode_length if labels is None else labels.size(1)

        batch_size = attrs.size(0)
        # hidden.size() should be (n_layers*n_directions, beam_size*batch_size, dim_hidden)
        hiddens = self.transform(attrs.float()).unsqueeze(0).unsqueeze(0)
        hiddens = hiddens.expand(self.n_layers*self.n_directions, beam_size, -1, -1)
        hiddens = hiddens.contiguous().view(-1, beam_size*batch_size, self.dim_hidden)
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        # last_output.size() == (beam_size, batch_size)
        last_output = [last_output for _ in range(beam_size)]
        # logits.shape will be [seq_length, beam_size, batch_size, vocab_size]
        logits = []
        beam_probs = np.full((beam_size, batch_size), -math.inf)
        beam_probs[0, :] = 0.0
        # last_indices.shape will be [seq_length, batch_size, beam_size]
        last_indices = []
        output_ids = []
        for step in range(decode_length):
            curr_inputs = []
            for beam in range(beam_size):
                use_tf = False if step == 0 else random.random() < tf_ratio
                if use_tf:
                    curr_input = labels[:, step-1]
                else:
                    curr_input = last_output[beam].detach()

                if len(curr_input.size()) == 1:
                    # curr_input are ids
                    curr_input = self.embedding(curr_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)
                curr_inputs.append(curr_input)

            curr_inputs = torch.stack(curr_inputs, dim=0)
            # curr_inputs.size() == (beam_size, batch_size, 1, dim_embedding)
            curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            output, new_hiddens = self.rnn(curr_inputs, hiddens)
            output = self.linear(output.squeeze(1))
            output = output.view(beam_size, batch_size, -1)
            new_hiddens = new_hiddens.view(self.n_layers*self.n_directions, beam_size, batch_size, -1)
            probs = torch.log_softmax(output.detach(), dim=-1)
            # top_probs.size() == top_indices.size() == (beam_size, batch_size, k)
            top_probs, top_indices = torch.topk(probs, k=beam_size, dim=-1)
            top_probs = top_probs.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()
            last_index = []
            output_id = []
            for bid in range(batch_size):
                beam_prob = []
                for beam in range(beam_size):
                    beam_prob.extend([
                        (
                            beam,
                            top_indices[beam, bid, i],
                            beam_probs[beam][bid] + top_probs[beam, bid, i]
                        )
                        for i in range(beam_size)
                    ])
                topk = sorted(beam_prob, key=lambda x: x[2], reverse=True)[:beam_size]
                last_index.append([item[0] for item in topk])
                output_id.append([item[1] for item in topk])
                beam_probs[:, bid] = np.array([item[2] for item in topk])

            last_indices.append(last_index)
            output_ids.append(output_id)

            new_hiddens = new_hiddens.permute([2, 0, 1, 3]).split(split_size=1, dim=0)
            hiddens = torch.stack([
                new_hiddens[bid].squeeze(0).index_select(dim=1, index=torch.tensor(indices).to(new_hiddens[bid].device))
                for bid, indices in enumerate(last_index)
            ], dim=0).permute([1, 2, 0, 3]).contiguous().view(-1, beam_size*batch_size, self.dim_hidden)

            output = output.transpose(0, 1).split(split_size=1, dim=0)
            output = [
                output[bid].squeeze(0).index_select(dim=0, index=torch.tensor(indices).to(output[bid].device))
                for bid, indices in enumerate(last_index)
            ]
            logits.append(output)

            last_output = [
                torch.tensor(
                    [output_id[bid][beam] for bid in range(batch_size)],
                    dtype=torch.long, device=attrs.device
                )
                for beam in range(beam_size)
            ]

        last_indices = np.array(last_indices)
        output_ids = np.array(output_ids)
        # back-trace the beams to get outputs
        beam_outputs = []
        beam_logits = []
        beam_decisions = []
        for bid in range(batch_size):
            this_index = np.arange(beam_size)
            step_logits = []
            step_output_ids = []
            for step in range(decode_length-1, -1, -1):
                this_logits = logits[step][bid].index_select(dim=0, index=torch.from_numpy(this_index).to(logits[step][bid].device))
                step_logits.append(this_logits)
                step_output_ids.append(output_ids[step, bid, this_index])
                this_index = last_indices[step, bid, this_index]

            step_logits = torch.stack(step_logits[::-1], dim=0)
            step_outputs, step_decisions = self._st_onehot(step_logits, np.array(step_output_ids[::-1]), hard=st)
            beam_outputs.append(step_outputs)
            beam_logits.append(step_logits)
            beam_decisions.append(step_decisions)

        logits = torch.stack(beam_logits).transpose(1, 2)
        outputs = torch.stack(beam_outputs).transpose(1, 2)
        decisions = torch.stack(beam_decisions).transpose(1, 2)
        return logits, outputs, decisions

    def forward_greedy(self, attrs, bos_id, labels=None, sampling=False,
                       tf_ratio=0.5, max_decode_length=50, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
            outputs: shape [batch_size, seq_length, vocab_size]
                     output words as one-hot vectors
        """
        decode_length = max_decode_length if labels is None else labels.size(1)

        hidden = self.transform(attrs.float()).unsqueeze(0)
        hidden = hidden.expand(self.n_layers*self.n_directions, -1, -1).contiguous()
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        logits = []
        outputs = []
        for step in range(decode_length):
            use_tf = False if step == 0 else random.random() < tf_ratio
            if use_tf:
                curr_input = labels[:, step-1]
            else:
                curr_input = last_output.detach()

            if len(curr_input.size()) == 1:
                curr_input = self.embedding(curr_input).unsqueeze(1)
            else:
                curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)

            output, hidden = self.rnn(curr_input, hidden)
            output = self.linear(output.squeeze(1))
            logits.append(output)
            if sampling:
                last_output = F.gumbel_softmax(output, hard=True)
            else:
                last_output = self._st_softmax(output, hard=True, dim=-1)
            outputs.append(self._st_softmax(output, hard=st, dim=-1))

        logits = torch.stack(logits).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        return logits, outputs


class LMRNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(LMRNN, self).__init__(*args, **kwargs)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in LM model.")
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def forward(self, inputs):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
        """
        inputs = self.embedding(inputs)
        output, _ = self.rnn(inputs)
        logits = self.linear(output)
        return logits

"""
Borrowed from https://github.com/karpathy/pytorch-made/blob/master/made.py
"""
class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)



class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_score.masked_fill_(attn_mask, -1e9)

        attn_weights = nn.Softmax(dim=-1)(attn_score)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(attn)

        return output, attn_weights


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.relu(self.linear1(inputs))
        output = self.linear2(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)

        return ffn_outputs, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len=300, d_model=768, n_layers=12, n_heads=8, p_drop=0.1, d_ff=500, pad_id=0):
        super(TransformerEncoder, self).__init__()
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(seq_len + 1, d_model)  # (seq_len+1, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)

        return outputs

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / d_model)

        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)


class LMTrm(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super(LMTrm, self).__init__(*args, **kwargs)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in LM model.")
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def forward(self, inputs):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
        """
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)

        logits = self.linear(outputs)

        return logits


class NLGTrm(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super(NLGTrm, self).__init__(*args, **kwargs)

        self.transform = nn.Linear(self.attr_vocab_size, self.dim_hidden)
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def _st_softmax(self, logits, hard=False, dim=-1):
        y_soft = logits.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def _st_onehot(self, logits, indices, hard=True, dim=-1):
        y_soft = logits.softmax(dim)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(logits.device)
        if len(logits.size()) == len(indices.size()) + 1:
            indices = indices.unsqueeze(-1)
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        if hard:
            return y_hard - y_soft.detach() + y_soft, y_hard
        else:
            return y_soft, y_hard

    def forward(self, inputs, attrs=None, bos_id=None, labels=None,
                tf_ratio=0.5, max_decode_length=50, beam_size=5, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, beam_size, seq_length, vocab_size]
            outputs: shape [batch_size, beam_size, seq_length, vocab_size]
                     output words as one-hot vectors (maybe soft)
            decisions: shape [batch_size, beam_size, seq_length, vocab_size]
                       output words as one-hot vectors (hard)
        """
        if beam_size == 1:
            logits, outputs = self.forward_greedy(
                attrs, bos_id, labels,
                tf_ratio=tf_ratio, max_decode_length=max_decode_length,
                st=st
            )
            return logits.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1)

        decode_length = max_decode_length if labels is None else labels.size(1)

        batch_size = attrs.size(0)
        # hidden.size() should be (n_layers*n_directions, beam_size*batch_size, dim_hidden)
        hiddens = self.transform(attrs.float()).unsqueeze(0).unsqueeze(0)
        hiddens = hiddens.expand(self.n_layers*self.n_directions, beam_size, -1, -1)
        hiddens = hiddens.contiguous().view(-1, beam_size*batch_size, self.dim_hidden)
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        # last_output.size() == (beam_size, batch_size)
        last_output = [last_output for _ in range(beam_size)]
        # logits.shape will be [seq_length, beam_size, batch_size, vocab_size]
        logits = []
        beam_probs = np.full((beam_size, batch_size), -math.inf)
        beam_probs[0, :] = 0.0
        # last_indices.shape will be [seq_length, batch_size, beam_size]
        last_indices = []
        output_ids = []
        for step in range(decode_length):
            curr_inputs = []
            for beam in range(beam_size):
                use_tf = False if step == 0 else random.random() < tf_ratio
                if use_tf:
                    curr_input = labels[:, step-1]
                else:
                    curr_input = last_output[beam].detach()

                if len(curr_input.size()) == 1:
                    # curr_input are ids
                    curr_input = self.embedding(curr_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)
                curr_inputs.append(curr_input)

            curr_inputs = torch.stack(curr_inputs, dim=0)
            # curr_inputs.size() == (beam_size, batch_size, 1, dim_embedding)
            curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            output, new_hiddens = self.rnn(curr_inputs, hiddens)
            output = self.linear(output.squeeze(1))
            output = output.view(beam_size, batch_size, -1)
            new_hiddens = new_hiddens.view(self.n_layers*self.n_directions, beam_size, batch_size, -1)
            probs = torch.log_softmax(output.detach(), dim=-1)
            # top_probs.size() == top_indices.size() == (beam_size, batch_size, k)
            top_probs, top_indices = torch.topk(probs, k=beam_size, dim=-1)
            top_probs = top_probs.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()
            last_index = []
            output_id = []
            for bid in range(batch_size):
                beam_prob = []
                for beam in range(beam_size):
                    beam_prob.extend([
                        (
                            beam,
                            top_indices[beam, bid, i],
                            beam_probs[beam][bid] + top_probs[beam, bid, i]
                        )
                        for i in range(beam_size)
                    ])
                topk = sorted(beam_prob, key=lambda x: x[2], reverse=True)[:beam_size]
                last_index.append([item[0] for item in topk])
                output_id.append([item[1] for item in topk])
                beam_probs[:, bid] = np.array([item[2] for item in topk])

            last_indices.append(last_index)
            output_ids.append(output_id)

            new_hiddens = new_hiddens.permute([2, 0, 1, 3]).split(split_size=1, dim=0)
            hiddens = torch.stack([
                new_hiddens[bid].squeeze(0).index_select(dim=1, index=torch.tensor(indices).to(new_hiddens[bid].device))
                for bid, indices in enumerate(last_index)
            ], dim=0).permute([1, 2, 0, 3]).contiguous().view(-1, beam_size*batch_size, self.dim_hidden)

            output = output.transpose(0, 1).split(split_size=1, dim=0)
            output = [
                output[bid].squeeze(0).index_select(dim=0, index=torch.tensor(indices).to(output[bid].device))
                for bid, indices in enumerate(last_index)
            ]
            logits.append(output)

            last_output = [
                torch.tensor(
                    [output_id[bid][beam] for bid in range(batch_size)],
                    dtype=torch.long, device=attrs.device
                )
                for beam in range(beam_size)
            ]

        last_indices = np.array(last_indices)
        output_ids = np.array(output_ids)
        # back-trace the beams to get outputs
        beam_outputs = []
        beam_logits = []
        beam_decisions = []
        for bid in range(batch_size):
            this_index = np.arange(beam_size)
            step_logits = []
            step_output_ids = []
            for step in range(decode_length-1, -1, -1):
                this_logits = logits[step][bid].index_select(dim=0, index=torch.from_numpy(this_index).to(logits[step][bid].device))
                step_logits.append(this_logits)
                step_output_ids.append(output_ids[step, bid, this_index])
                this_index = last_indices[step, bid, this_index]

            step_logits = torch.stack(step_logits[::-1], dim=0)
            step_outputs, step_decisions = self._st_onehot(step_logits, np.array(step_output_ids[::-1]), hard=st)
            beam_outputs.append(step_outputs)
            beam_logits.append(step_logits)
            beam_decisions.append(step_decisions)

        logits = torch.stack(beam_logits).transpose(1, 2)
        outputs = torch.stack(beam_outputs).transpose(1, 2)
        decisions = torch.stack(beam_decisions).transpose(1, 2)
        return logits, outputs, decisions

    def forward_greedy(self, attrs, bos_id, labels=None, sampling=False,
                       tf_ratio=0.5, max_decode_length=50, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
            outputs: shape [batch_size, seq_length, vocab_size]
                     output words as one-hot vectors
        """
        decode_length = max_decode_length if labels is None else labels.size(1)

        hidden = self.transform(attrs.float()).unsqueeze(0)
        hidden = hidden.expand(self.n_layers*self.n_directions, -1, -1).contiguous()
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        logits = []
        outputs = []
        for step in range(decode_length):
            use_tf = False if step == 0 else random.random() < tf_ratio
            if use_tf:
                curr_input = labels[:, step-1]
            else:
                curr_input = last_output.detach()

            if len(curr_input.size()) == 1:
                curr_input = self.embedding(curr_input).unsqueeze(1)
            else:
                curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)

            output, hidden = self.rnn(curr_input, hidden)
            output = self.linear(output.squeeze(1))
            logits.append(output)
            if sampling:
                last_output = F.gumbel_softmax(output, hard=True)
            else:
                last_output = self._st_softmax(output, hard=True, dim=-1)
            outputs.append(self._st_softmax(output, hard=st, dim=-1))

        logits = torch.stack(logits).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        return logits, outputs


class DTTreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        """
        super(DTTreeGRU, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        # linear parameters for transformation from input to hidden state. same for all 5 gates
        self.gate_ih = nn.Linear(in_features=input_size, out_features=5 * hidden_size, bias=True)
        self.gate_lhh = nn.Linear(in_features=hidden_size, out_features=5 * hidden_size, bias=False)
        self.gate_rhh = nn.Linear(in_features=hidden_size, out_features=5 * hidden_size, bias=False)
        self.cell_ih = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.cell_lhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.cell_rhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        # self.reset_parameters()

    def reset_parameters(self):
        weight_ih = block_orth_normal_initializer([self._input_size, ], [self._hidden_size] * 5)
        self.gate_ih.weight.data.copy_(weight_ih)
        nn.init.constant(self.gate_ih.bias, 0.0)

        weight_lhh = block_orth_normal_initializer([self._input_size, ], [self._hidden_size] * 5)
        self.gate_lhh.weight.data.copy_(weight_lhh)

        weight_rhh = block_orth_normal_initializer([self._input_size, ], [self._hidden_size] * 5)
        self.gate_rhh.weight.data.copy_(weight_rhh)

        nn.init.orthogonal(self.cell_ih.weight)
        nn.init.constant(self.cell_ih.bias, 0.0)
        nn.init.orthogonal(self.cell_lhh.weight)
        nn.init.orthogonal(self.cell_rhh.weight)

    def forward(self, inputs, indexes, trees):
        """
        :param inputs: batch first
        :param tree:
        :return: output, h_n
        """

        max_length, batch_size, input_dim = inputs.size()
        dt_state = []
        for b in range(batch_size):
            dt_state.append({})

        for step in range(max_length):
            step_inputs, left_child_hs, right_child_hs = [], [], []
            for b, tree in enumerate(trees):
                index = indexes[step, b]
                if index == -1:
                    continue
                step_inputs.append(inputs[index, b])
                if tree[index].left_num == 0:
                    left_child_h = Var(inputs.data.new(self._hidden_size).fill_(0.))
                else:
                    left_child_h = [dt_state[b][child.index] for child in tree[index].left_children]
                    left_child_h = torch.stack(left_child_h, 0)
                    left_child_h = torch.mean(left_child_h, dim=0)  # sum @mszhang

                if tree[index].right_num == 0:
                    right_child_h = Var(inputs.data.new(self._hidden_size).fill_(0.))
                else:
                    right_child_h = [dt_state[b][child.index] for child in tree[index].right_children]
                    right_child_h = torch.stack(right_child_h, 0)
                    right_child_h = torch.mean(right_child_h, dim=0)  # sum @mszhang

                left_child_hs.append(left_child_h)
                right_child_hs.append(right_child_h)

            step_inputs = torch.stack(step_inputs, 0)
            left_child_hs = torch.stack(left_child_hs, 0)
            right_child_hs = torch.stack(right_child_hs, 0)

            results = self.node_forward(step_inputs, left_child_hs, right_child_hs)

            results_count = 0
            for b in range(batch_size):  # collect the current step results
                index = indexes[step, b]
                if index == -1:
                    continue
                dt_state[b][index] = results[results_count]
                results_count += 1

        outputs, output_t = [], []
        for b, length in enumerate([len(tree) for tree in trees]):
            output = [dt_state[b][idx] for idx in range(0, length)]
            output.extend([Var(inputs.data.new(self._hidden_size).fill_(0.))] * (max_length - length))
            outputs.append(torch.stack(output, 0))
            output_t.append(Var(inputs.data.new(self._hidden_size).fill_(0.)))

        return torch.stack(outputs, 0), torch.stack(output_t, 0)

    def forward_v2(self, inputs, indexes, trees):
        """
        :param inputs: batch first
        :param tree:
        :return: output, h_n
        """
        max_length, batch_size, input_dim = inputs.size()
        dt_state = []
        degree = np.zeros((batch_size, max_length), dtype=np.int32)
        last_indexes = np.zeros((batch_size), dtype=np.int32)
        for b, tree in enumerate(trees):
            dt_state.append({})
            for index in range(max_length):
                degree[b, index] = tree[index].left_num + tree[index].right_num

        for step in range(max_length):
            step_inputs, left_child_hs, right_child_hs, compute_indexes = [], [], [], []
            for b, tree in enumerate(trees):
                last_index = last_indexes[b]
                for idx in range(last_index, max_length):
                    cur_index = indexes[idx, b]
                    if degree[b, cur_index] > 0:
                        break
                    last_indexes[b] += 1
                    compute_indexes.append((b, cur_index))
                    step_inputs.append(inputs[cur_index, b])
                    if tree[cur_index].left_num == 0:
                        left_child_h = Var(inputs.data.new(self._hidden_size).fill_(0.))
                    else:
                        left_child_h = [dt_state[b][child.index] for child in tree[cur_index].left_children]
                        left_child_h = torch.stack(left_child_h, 0)
                        left_child_h = torch.sum(left_child_h, dim=0)

                    if tree[cur_index].right_num == 0:
                        right_child_h = Var(inputs.data.new(self._hidden_size).fill_(0.))
                    else:
                        right_child_h = [dt_state[b][child.index] for child in tree[cur_index].right_children]
                        right_child_h = torch.stack(right_child_h, 0)
                        right_child_h = torch.sum(right_child_h, dim=0)

                    left_child_hs.append(left_child_h)
                    right_child_hs.append(right_child_h)

            if len(compute_indexes) == 0:
                for last_index in last_indexes:
                    if last_index != max_length:
                        print('bug exists: some nodes are not completed')
                break

            step_inputs = torch.stack(step_inputs, 0)
            left_child_hs = torch.stack(left_child_hs, 0)
            right_child_hs = torch.stack(right_child_hs, 0)

            results = self.node_forward(step_inputs, left_child_hs, right_child_hs)
            for idx, (b, cur_index) in enumerate(compute_indexes):
                dt_state[b][cur_index] = results[idx]
                if trees[b][cur_index].parent is not None:
                    parent_index = trees[b][cur_index].parent.index
                    degree[b, parent_index] -= 1
                    if degree[b, parent_index] < 0:
                        print('strange bug')

        outputs, output_t = [], []
        for b in range(batch_size):
            output = [dt_state[b][idx] for idx in range(1, max_length)] + [dt_state[b][0]]  # 1 mszhang, 0 kiro
            outputs.append(torch.stack(output, 0))
            output_t.append(dt_state[b][0])

        return torch.stack(outputs, 0), torch.stack(output_t, 0)

    def node_forward(self, input, left_child_h, right_child_h):
        gates = self.gate_ih(input) + self.gate_lhh(left_child_h) + self.gate_rhh(right_child_h)
        gates = torch.sigmoid(gates)
        rl, rr, zl, zr, z = torch.split(gates, gates.size(1) // 5, dim=1)

        gated_l, gated_r = rl * left_child_h, rr * right_child_h
        cell = self.cell_ih(input) + self.cell_lhh(gated_l) + self.cell_rhh(gated_r)
        cell = torch.tanh(cell)

        hidden = zl * left_child_h + zr * right_child_h + z * cell

        return hidden

    def highway_node_forward(self, input, left_child_h, right_child_h):
        _x = self.gate_ih(input)
        gates = _x[:, :self._hidden_size * 6] + self.gate_lhh(left_child_h) + self.gate_rhh(right_child_h)
        gates = torch.sigmoid(gates)
        rl, rr, zl, zr, z, r = gates.chunk(chunks=6, dim=1)

        gated_l, gated_r = rl * left_child_h, rr * right_child_h
        cell = self.cell_ih(input) + self.cell_lhh(gated_l) + self.cell_rhh(gated_r)
        cell = torch.tanh(cell)

        hidden = zl * left_child_h + zr * right_child_h + z * cell

        _k = _x[:, self._hidden_size * 6:]
        hidden = r * hidden + (1.0 - r) * _k
        """hidden = torch.stack([input, left_child_h, right_child_h], dim=0)
        hidden, _ = torch.max(hidden, dim=0)"""
        return hidden


class TDTreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        """
        super(TDTreeGRU, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        # linear parameters for transformation from input to hidden state. same for all 5 gates
        self.gate_ih = nn.Linear(in_features=input_size, out_features=3 * hidden_size, bias=True)
        self.gate_lhh = nn.Linear(in_features=hidden_size, out_features=3 * hidden_size, bias=False)
        self.gate_rhh = nn.Linear(in_features=hidden_size, out_features=3 * hidden_size, bias=False)
        self.cell_ih = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.cell_lhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.cell_rhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        # self.reset_parameters()

    def reset_parameters(self):
        weight_ih = block_orth_normal_initializer([self._input_size, ], [self._hidden_size] * 3)
        self.gate_ih.weight.data.copy_(weight_ih)
        nn.init.constant(self.gate_ih.bias, 0.0)

        weight_lhh = block_orth_normal_initializer([self._input_size, ], [self._hidden_size] * 3)
        self.gate_lhh.weight.data.copy_(weight_lhh)

        weight_rhh = block_orth_normal_initializer([self._input_size, ], [self._hidden_size] * 3)
        self.gate_rhh.weight.data.copy_(weight_rhh)

        nn.init.orthogonal(self.cell_ih.weight)
        nn.init.constant(self.cell_ih.bias, 0.0)
        nn.init.orthogonal(self.cell_lhh.weight)
        nn.init.orthogonal(self.cell_rhh.weight)

    def forward(self, inputs, indexes, trees):
        """
        :param inputs:
        :param tree:
        :return: output, h_n
        """
        max_length, batch_size, input_dim = inputs.size()
        td_state = []
        for b in range(batch_size):
            td_state.append({})

        for step in reversed(range(max_length)):
            step_inputs, left_parent_hs, right_parent_hs = [], [], []
            for b, tree in enumerate(trees):
                index = indexes[step, b]
                if index == -1:
                    continue
                step_inputs.append(inputs[index, b])
                parent_h = Var(inputs[0].data.new(self._hidden_size).fill_(0.))
                if tree[index].parent is None:
                    left_parent_hs.append(parent_h)
                    right_parent_hs.append(parent_h)
                else:
                    valid_parent_h = td_state[b][tree[index].parent.index]
                    if tree[index].is_left:
                        left_parent_hs.append(valid_parent_h)
                        right_parent_hs.append(parent_h)
                    else:
                        left_parent_hs.append(parent_h)
                        right_parent_hs.append(valid_parent_h)

            step_inputs = torch.stack(step_inputs, 0)
            left_parent_hs = torch.stack(left_parent_hs, 0)
            right_parent_hs = torch.stack(right_parent_hs, 0)

            results = self.node_forward(step_inputs, left_parent_hs, right_parent_hs)

            result_count = 0
            for b in range(batch_size):
                index = indexes[step, b]
                if index == -1:
                    continue
                td_state[b][index] = results[result_count]
                result_count += 1

        outputs, output_t = [], []
        for b, length in enumerate([len(tree) for tree in trees]):
            output = [td_state[b][idx] for idx in range(0, length)]
            output.extend([Var(inputs.data.new(self._hidden_size).fill_(0.))] * (max_length - length))
            outputs.append(torch.stack(output, 0))
            output_t.append(Var(inputs.data.new(self._hidden_size).fill_(0.)))

        return torch.stack(outputs, 0), torch.stack(output_t, 0)

    def forward_v2(self, inputs, indexes, trees):
        """
        :param inputs:
        :param tree:
        :return: output, h_n
        """
        max_length, batch_size, input_dim = inputs.size()
        degree = np.ones((batch_size, max_length), dtype=np.int32)
        last_indexes = max_length * np.ones((batch_size), dtype=np.int32)
        td_state = []
        for b in range(batch_size):
            td_state.append({})
            root_index = indexes[max_length - 1, b]
            degree[b, root_index] = 0

        for step in range(max_length):
            step_inputs, left_parent_hs, right_parent_hs, compute_indexes = [], [], [], []
            for b, tree in enumerate(trees):
                last_index = last_indexes[b]
                for idx in reversed(range(last_index)):
                    cur_index = indexes[idx, b]
                    if degree[b, cur_index] > 0:
                        break
                    last_indexes[b] -= 1
                    compute_indexes.append((b, cur_index))
                    step_inputs.append(inputs[cur_index, b])
                    parent_h = Var(inputs[0].data.new(self._hidden_size).fill_(0.))
                    if tree[cur_index].parent is None:
                        left_parent_hs.append(parent_h)
                        right_parent_hs.append(parent_h)
                    else:
                        valid_parent_h = td_state[b][tree[cur_index].parent.index]
                        if tree[cur_index].is_left:
                            left_parent_hs.append(valid_parent_h)
                            right_parent_hs.append(parent_h)
                        else:
                            left_parent_hs.append(parent_h)
                            right_parent_hs.append(valid_parent_h)

            if len(compute_indexes) == 0:
                for last_index in last_indexes:
                    if last_index != 0:
                        print('bug exists: some nodes are not completed')
                break

            step_inputs = torch.stack(step_inputs, 0)
            left_parent_hs = torch.stack(left_parent_hs, 0)
            right_parent_hs = torch.stack(right_parent_hs, 0)

            results = self.node_forward(step_inputs, left_parent_hs, right_parent_hs)
            for idx, (b, cur_index) in enumerate(compute_indexes):
                td_state[b][cur_index] = results[idx]
                for child in trees[b][cur_index].left_children:
                    degree[b, child.index] -= 1
                    if degree[b, child.index] < 0:
                        print('strange bug')
                for child in trees[b][cur_index].right_children:
                    degree[b, child.index] -= 1
                    if degree[b, child.index] < 0:
                        print('strange bug')

        outputs, output_t = [], []
        for b in range(batch_size):
            output = [td_state[b][idx] for idx in range(1, max_length)] + [td_state[b][0]]  # modified by kiro
            outputs.append(torch.stack(output, 0))
            output_t.append(td_state[b][0])

        return torch.stack(outputs, 0), torch.stack(output_t, 0)

    def node_forward(self, input, left_parent_h, right_parent_h):
        gates = self.gate_ih(input) + self.gate_lhh(left_parent_h) + self.gate_rhh(right_parent_h)
        gates = torch.sigmoid(gates)
        rp, zp, z = torch.split(gates, gates.size(1) // 3, dim=1)

        gated_l, gated_r = rp * left_parent_h, rp * right_parent_h

        cell = self.cell_ih(input) + self.cell_lhh(gated_l) + self.cell_rhh(gated_r)
        cell = torch.tanh(cell)

        hidden = zp * (left_parent_h + right_parent_h) + z * cell

        return hidden

    def highway_node_forward(self, input, left_parent_h, right_parent_h):
        _x = self.gate_ih(input)
        gates = _x[:, :self._hidden_size * 4] + self.gate_lhh(left_parent_h) + self.gate_rhh(right_parent_h)
        gates = torch.sigmoid(gates)
        rp, zp, z, r = gates.chunk(chunks=4, dim=1)

        gated_l, gated_r = rp * left_parent_h, rp * right_parent_h

        cell = self.cell_ih(input) + self.cell_lhh(gated_l) + self.cell_rhh(gated_r)
        cell = torch.tanh(cell)

        hidden = zp * (left_parent_h + right_parent_h) + z * cell

        _k = _x[:, self._hidden_size * 4:]
        hidden = r * hidden + (1.0 - r) * _k
        """hidden = torch.stack([input, left_parent_h, right_parent_h], dim=0)
        hidden, _ = torch.max(hidden, dim=0)"""
        return hidden


class TreeGRUEncoder(nn.Module):
    """ The standard RNN encoder.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(TreeGRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        """self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True)  # batch_first = False
        self.transform = nn.Linear(in_features=2*hidden_size, out_features=input_size, bias=True)"""
        self.dt_tree = DTTreeGRU(input_size, hidden_size)
        self.td_tree = TDTreeGRU(input_size, hidden_size)

    def forward(self, input, heads, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns.
        inputs: [L, B, H], including the -ROOT-
        heads: [heads] * B
        """
        """emb = self.dropout(input)

        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None:
            outputs = unpack(outputs)[0]

        outputs = self.dropout(self.transform(outputs))"""
        outputs = input  # @kiro
        max_length, batch_size, input_dim = outputs.size()
        trees = []
        indexes = np.full((max_length, batch_size), -1, dtype=np.int32)  # a col is a sentence
        for b, head in enumerate(heads):
            root, tree = creatTree(head)  # head: a sentence's heads; sentence base
            root.traverse()  # traverse the tree
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs, dt_hidden_ts = self.dt_tree.forward(outputs, indexes, trees)
        td_outputs, td_hidden_ts = self.td_tree.forward(outputs, indexes, trees)

        outputs = torch.cat([dt_outputs, td_outputs], dim=2).transpose(0, 1)
        output_t = torch.cat([dt_hidden_ts, td_hidden_ts], dim=1).unsqueeze(0)

        return outputs, output_t



class NLGTreeTrm(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super(NLGTreeTrm, self).__init__(*args, **kwargs)
        self.treeEncoder = TreeGRUEncoder(self.dim_hidden, self.dim_hidden)
        self.transform = nn.Linear(self.attr_vocab_size, self.dim_hidden)
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def _st_softmax(self, logits, hard=False, dim=-1):
        y_soft = logits.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def _st_onehot(self, logits, indices, hard=True, dim=-1):
        y_soft = logits.softmax(dim)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(logits.device)
        if len(logits.size()) == len(indices.size()) + 1:
            indices = indices.unsqueeze(-1)
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        if hard:
            return y_hard - y_soft.detach() + y_soft, y_hard
        else:
            return y_soft, y_hard

    def RoI_pass(self, inputs):
        x_RoI = inputs.transpose(0, 1)
        x_RoI, _ = self.treeEncoder(x_RoI, self.head)
        x_RoI = x_RoI.transpose(0, 1)
        x_RoI = self.tree_projection(x_RoI)

        return torch.cat((inputs, x_RoI), 2)


    def forward(self, inputs, attrs=None, bos_id=None, labels=None,
                tf_ratio=0.5, max_decode_length=50, beam_size=5, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, beam_size, seq_length, vocab_size]
            outputs: shape [batch_size, beam_size, seq_length, vocab_size]
                     output words as one-hot vectors (maybe soft)
            decisions: shape [batch_size, beam_size, seq_length, vocab_size]
                       output words as one-hot vectors (hard)
        """
        RoIs = self.RoI_pass(inputs)
        if beam_size == 1:
            logits, outputs = self.forward_greedy(
                attrs, bos_id, labels,
                tf_ratio=tf_ratio, max_decode_length=max_decode_length,
                st=st
            )
            return logits.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1)

        decode_length = max_decode_length if labels is None else labels.size(1)

        batch_size = attrs.size(0)
        # hidden.size() should be (n_layers*n_directions, beam_size*batch_size, dim_hidden)
        hiddens = self.transform(attrs.float()).unsqueeze(0).unsqueeze(0)
        hiddens = hiddens.expand(self.n_layers*self.n_directions, beam_size, -1, -1)
        hiddens = hiddens.contiguous().view(-1, beam_size*batch_size, self.dim_hidden)
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        # last_output.size() == (beam_size, batch_size)
        last_output = [last_output for _ in range(beam_size)]
        # logits.shape will be [seq_length, beam_size, batch_size, vocab_size]
        logits = []
        beam_probs = np.full((beam_size, batch_size), -math.inf)
        beam_probs[0, :] = 0.0
        # last_indices.shape will be [seq_length, batch_size, beam_size]
        last_indices = []
        output_ids = []
        for step in range(decode_length):
            curr_inputs = []
            for beam in range(beam_size):
                use_tf = False if step == 0 else random.random() < tf_ratio
                if use_tf:
                    curr_input = labels[:, step-1]
                else:
                    curr_input = last_output[beam].detach()

                if len(curr_input.size()) == 1:
                    # curr_input are ids
                    curr_input = self.embedding(curr_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)
                curr_inputs.append(curr_input)

            curr_inputs = torch.stack(curr_inputs, dim=0)
            # curr_inputs.size() == (beam_size, batch_size, 1, dim_embedding)
            curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            output, new_hiddens = self.rnn(curr_inputs, hiddens)
            output = self.linear(output.squeeze(1))
            output = output.view(beam_size, batch_size, -1)
            new_hiddens = new_hiddens.view(self.n_layers*self.n_directions, beam_size, batch_size, -1)
            probs = torch.log_softmax(output.detach(), dim=-1)
            # top_probs.size() == top_indices.size() == (beam_size, batch_size, k)
            top_probs, top_indices = torch.topk(probs, k=beam_size, dim=-1)
            top_probs = top_probs.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()
            last_index = []
            output_id = []
            for bid in range(batch_size):
                beam_prob = []
                for beam in range(beam_size):
                    beam_prob.extend([
                        (
                            beam,
                            top_indices[beam, bid, i],
                            beam_probs[beam][bid] + top_probs[beam, bid, i]
                        )
                        for i in range(beam_size)
                    ])
                topk = sorted(beam_prob, key=lambda x: x[2], reverse=True)[:beam_size]
                last_index.append([item[0] for item in topk])
                output_id.append([item[1] for item in topk])
                beam_probs[:, bid] = np.array([item[2] for item in topk])

            last_indices.append(last_index)
            output_ids.append(output_id)

            new_hiddens = new_hiddens.permute([2, 0, 1, 3]).split(split_size=1, dim=0)
            hiddens = torch.stack([
                new_hiddens[bid].squeeze(0).index_select(dim=1, index=torch.tensor(indices).to(new_hiddens[bid].device))
                for bid, indices in enumerate(last_index)
            ], dim=0).permute([1, 2, 0, 3]).contiguous().view(-1, beam_size*batch_size, self.dim_hidden)

            output = output.transpose(0, 1).split(split_size=1, dim=0)
            output = [
                output[bid].squeeze(0).index_select(dim=0, index=torch.tensor(indices).to(output[bid].device))
                for bid, indices in enumerate(last_index)
            ]
            logits.append(output)

            last_output = [
                torch.tensor(
                    [output_id[bid][beam] for bid in range(batch_size)],
                    dtype=torch.long, device=attrs.device
                )
                for beam in range(beam_size)
            ]

        last_indices = np.array(last_indices)
        output_ids = np.array(output_ids)
        # back-trace the beams to get outputs
        beam_outputs = []
        beam_logits = []
        beam_decisions = []
        for bid in range(batch_size):
            this_index = np.arange(beam_size)
            step_logits = []
            step_output_ids = []
            for step in range(decode_length-1, -1, -1):
                this_logits = logits[step][bid].index_select(dim=0, index=torch.from_numpy(this_index).to(logits[step][bid].device))
                step_logits.append(this_logits)
                step_output_ids.append(output_ids[step, bid, this_index])
                this_index = last_indices[step, bid, this_index]

            step_logits = torch.stack(step_logits[::-1], dim=0)
            step_outputs, step_decisions = self._st_onehot(step_logits, np.array(step_output_ids[::-1]), hard=st)
            beam_outputs.append(step_outputs)
            beam_logits.append(step_logits)
            beam_decisions.append(step_decisions)

        logits = torch.stack(beam_logits).transpose(1, 2)
        outputs = torch.stack(beam_outputs).transpose(1, 2)
        decisions = torch.stack(beam_decisions).transpose(1, 2)
        return logits, outputs, decisions, RoIs

    def forward_greedy(self, attrs, bos_id, labels=None, sampling=False,
                       tf_ratio=0.5, max_decode_length=50, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
            outputs: shape [batch_size, seq_length, vocab_size]
                     output words as one-hot vectors
        """
        decode_length = max_decode_length if labels is None else labels.size(1)

        hidden = self.transform(attrs.float()).unsqueeze(0)
        hidden = hidden.expand(self.n_layers*self.n_directions, -1, -1).contiguous()
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        logits = []
        outputs = []
        for step in range(decode_length):
            use_tf = False if step == 0 else random.random() < tf_ratio
            if use_tf:
                curr_input = labels[:, step-1]
            else:
                curr_input = last_output.detach()

            if len(curr_input.size()) == 1:
                curr_input = self.embedding(curr_input).unsqueeze(1)
            else:
                curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)

            output, hidden = self.rnn(curr_input, hidden)
            output = self.linear(output.squeeze(1))
            logits.append(output)
            if sampling:
                last_output = F.gumbel_softmax(output, hard=True)
            else:
                last_output = self._st_softmax(output, hard=True, dim=-1)
            outputs.append(self._st_softmax(output, hard=st, dim=-1))

        logits = torch.stack(logits).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        return logits, outputs



class SMDLCriterion(nn.Module):
    def __init__(self, loss_weight, pretrain_epochs=0,
                 LM=None, LM2=None, lambda_xy=0.1, lambda_yx=0.1,
                 made_n_samples=1, propagate_other=False):
        super(SMDLCriterion, self).__init__()
        self.pretrain_epochs = pretrain_epochs
        self.epoch = 0
        self.propagate_other = propagate_other
        self.lambda_xy = lambda_xy
        self.lambda_yx = lambda_yx
        self.LM = LM
        self.LM2 = LM2
        if LM is None:
            raise ValueError("Language model not provided")
        if LM2 is None:
            raise ValueError("Language model v2 not provided")

        self.made_n_samples = made_n_samples
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')
        self.CE = nn.CrossEntropyLoss(weight=loss_weight, reduction='sum')

    def get_log_joint_prob_nlg(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, seq_length, vocab_size]
            decisions: tensor of shape [batch_size, seq_length, vocab_size]
                       one-hot vector of decoded word-ids
        returns:
            log_joint_prob: tensor of shape [batch_size]
        """
        probs = torch.softmax(logits, dim=-1)
        return (decisions * probs).sum(dim=-1).log().sum(dim=-1)

    def get_log_joint_prob_nlu(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, attr_vocab_size]
            decisions: tensor of shape [batch_size, attr_vocab_size]
                       decisions(0/1)
        returns:
            log_joint_prob: tensor of shape [batch_size]
        """
        probs = torch.sigmoid(logits)
        decisions = decisions.float()
        probs = probs * decisions + (1-probs) * (1-decisions)
        return probs.log().sum(dim=-1)

    def epoch_end(self):
        self.epoch += 1
        if self.epoch == self.pretrain_epochs:
            print_time_info("pretrain finished, starting using duality loss")

    def get_scheduled_loss(self, dual_loss):
        if self.epoch < self.pretrain_epochs:
            return torch.tensor(0.0)
        return dual_loss

    def forward(self, nlg1_logits, nlg1_outputs, nlg2_logits, nlg1_targets, nlg2_targets):
        """
        args:
            nlg_logits: tensor of shape [batch_size, seq_length, vocab_size]
            nlg_outputs: tensor of shape [batch_size, seq_length, vocab_size]
            nlg_targets: tensor of shape [batch_size, seq_length]
            nlg2_logits: tensor of shape [batch_size, attr_vocab_size]
            nlg2_targets: tensor of shape [batch_size, attr_vocab_size]
        """
        nlg1_logits_1d = nlg1_logits.contiguous().view(-1, nlg1_logits.size(-1))
        nlg1_targets_1d = nlg1_targets.contiguous().view(-1)
        nlg1_sup_loss = self.CE(nlg1_logits_1d, nlg1_targets_1d)
        nlg2_sup_loss = self.BCE(nlg2_logits, nlg2_targets)

        log_p_x = self.LM.get_log_prob(nlg1_targets)
        log_p_y = self.LM.get_log_prob(nlg2_targets)

        log_p_y_x = self.get_log_joint_prob_nlg(nlg1_logits, nlg1_outputs)
        nlg2_decisions = (nlg2_logits.sigmoid() >= 0.5).float()
        log_p_x_y = self.get_log_joint_prob_nlg(nlg2_logits, nlg2_decisions)

        if self.propagate_other:
            nlg1_loss_dual = (log_p_x + log_p_y_x - log_p_y - log_p_x_y).pow(2).mean()
            nlg2_loss_dual = (log_p_x + log_p_y_x - log_p_y - log_p_x_y).pow(2).mean()
        else:
            nlg1_loss_dual = (log_p_x + log_p_y_x - log_p_y - log_p_x_y.detach()).pow(2).mean()
            nlg2_loss_dual = (log_p_x + log_p_y_x.detach() - log_p_y - log_p_x_y).pow(2).mean()

        nlg1_loss_dual = self.lambda_xy * self.get_scheduled_loss(nlg1_loss_dual)
        nlg2_loss_dual = self.lambda_yx * self.get_scheduled_loss(nlg2_loss_dual)

        return nlg1_sup_loss + nlg2_sup_loss, nlg1_loss_dual + nlg2_loss_dual


class RoIFilter(nn.Module):
    def __init__(self, d_model, S):
        super().__init__()
        self.mk=nn.Linear(d_model, S, bias=False)
        self.mv=nn.Linear(S, d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, threshold_omega):
        attn=self.mk(queries)
        attn=self.softmax(attn)
        attn=attn/torch.sum(attn, dim=2,keepdim=True)
        out=self.mv(attn)
        scores = nn.Sigmoid(out)

        valid_queries_id = []
        for i, score in enumerate(scores):
            if score >= threshold_omega:
                valid_queries_id.append(i)
        valid_queries_id = torch.LongTensor(valid_queries_id)
        valid_queries = torch.gather(queries, dim=1, index=valid_queries_id)
        return valid_queries


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features1, features2, labels=None, mask=None):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features1.is_cuda
                  else torch.device('cpu'))

        if len(features1.shape) < 3 or len(features2.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features1.shape) > 3 or len(features2.shape) > 3:
            features1 = features1.view(features1.shape[0], features1.shape[1], -1)
            features2 = features2.view(features2.shape[0], features2.shape[1], -1)

        batch_size = features1.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features1.shape[1]
        contrast_feature = torch.cat(torch.unbind(features1, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features1[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        contrast_count2 = features2.shape[1]
        contrast_feature2 = torch.cat(torch.unbind(features2, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature2 = features2[:, 0]
            anchor_count2 = 1
        elif self.contrast_mode == 'all':
            anchor_feature2 = contrast_feature2
            anchor_count2 = contrast_count2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_dot_contrast2 = torch.div(
            torch.matmul(anchor_feature2, contrast_feature2.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast+anchor_dot_contrast2, dim=1, keepdim=True)
        logits = anchor_dot_contrast+anchor_dot_contrast2 - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count+anchor_count2, contrast_count+contrast_count2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    assert isinstance(scores, Variable)
    shape = scores.size()
    assert len(shape) == 1
    increment = torch.ones(shape)
    increment[oracle_index] = 0
    return scores + Variable(increment)

class Feedforward(nn.Sequential):

    def __init__(self, input_dim, hidden_dims, output_dim):
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i, (prev_dim, next_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(prev_dim, next_dim))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        super(Feedforward, self).__init__(*layers)


class TopDownParser(nn.Module):
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
    ):
        super(TopDownParser, self).__init__()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = nn.Embedding(tag_vocab.size, tag_embedding_dim)
        self.word_embeddings = nn.Embedding(word_vocab.size, word_embedding_dim)

        self.lstm = nn.LSTM(
            input_size = tag_embedding_dim + word_embedding_dim,
            hidden_size = lstm_dim,
            num_layers = lstm_layers,
            dropout = dropout,
            bidirectional = True)

        self.f_label = Feedforward(
            2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(
            2 * lstm_dim, [split_hidden_dim], 1)


    def parse(self, sentence, gold=None, explore=True):
        is_train = gold is not None
        assert is_train == self.training

        indexes = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_index = self.tag_vocab.index(tag)
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_index = self.word_vocab.index(word)
            indexes.append([tag_index, word_index])

        indexes = torch.LongTensor(indexes).t()
        self.sentence = sentence
        self.gold = gold
        self.explore = explore
        loss = self(indexes)
        return self.tree, loss

    def forward(self, indexes):
        sentence = self.sentence
        gold = self.gold
        explore = self.explore
        is_train = self.training

        embeddings = torch.cat([self.tag_embeddings(indexes[0]),
                                self.word_embeddings(indexes[1])],
                                -1)
        lstm_outputs, _ = self.lstm(embeddings.unsqueeze(1))
        lstm_outputs = lstm_outputs.squeeze()

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            # return dy.concatenate([forward, backward])
            return torch.cat([forward, backward])

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))

            if is_train:
                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index)

            # label_scores_np = label_scores.npvalue()
            label_scores_np = label_scores.data.numpy()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1)
            argmax_label = self.label_vocab.value(argmax_label_index)

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    # if argmax_label != oracle_label else dy.zeros(1))
                    if argmax_label != oracle_label else Variable(torch.zeros(1)))
            else:
                label = argmax_label
                label_loss = label_scores[argmax_label_index]

            if right - left == 1:
                tag, word = sentence[left]
                tree = trees.LeafParseNode(left, tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right):
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))
            # left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            # right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
            left_scores = self.f_split(torch.stack(left_encodings))
            right_scores = self.f_split(torch.stack(right_encodings))
            split_scores = left_scores + right_scores
            # split_scores = dy.reshape(split_scores, (len(left_encodings),))
            split_scores = split_scores.view(len(left_encodings))

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            # split_scores_np = split_scores.npvalue()
            split_scores_np = split_scores.data.numpy()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    # if argmax_split != oracle_split else dy.zeros(1))
                    if argmax_split != oracle_split else Variable(torch.zeros(1)))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        # return tree, loss
        self.tree = tree
        return loss

