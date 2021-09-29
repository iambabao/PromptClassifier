# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from src.models.utils import ce_loss


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.start_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.end_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            length=None,
            labels=None,
    ):
        batch_size = input_ids.shape[0]
        max_seq_length = input_ids.shape[1]
        indexes = torch.arange(max_seq_length).expand(batch_size, max_seq_length).to(self.device)

        # mask for real tokens with shape (batch_size, max_seq_length)
        token_mask = torch.less(indexes, length.unsqueeze(-1))
        # mask for valid spans with shape (batch_size, max_seq_length, max_seq_length)
        span_mask = torch.logical_and(
            token_mask.unsqueeze(-1).expand(-1, -1, max_seq_length),
            token_mask.unsqueeze(-2).expand(-1, max_seq_length, -1),
        ).triu()

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)

        start_expanded = self.start_layer(sequence_output).unsqueeze(2).expand(-1, -1, max_seq_length, -1)
        end_expanded = self.end_layer(sequence_output).unsqueeze(1).expand(-1, max_seq_length, -1, -1)
        span_matrix = torch.cat([start_expanded, end_expanded], dim=-1)
        logits = self.output_layer(span_matrix)  # (batch_size, max_num_tokens, max_num_tokens, 2)
        outputs = (span_mask.unsqueeze(-1) * logits,) + outputs

        if labels is not None:
            labels = labels.view(-1, max_seq_length, max_seq_length)
            loss = ce_loss(logits, labels, span_mask)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
