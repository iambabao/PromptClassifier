# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            label=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        pooled_output = outputs[1]  # (batch_size, hidden_size)
        logits = self.output_layer(pooled_output)
        outputs = (logits,) + outputs

        if label is not None:
            loss = F.cross_entropy(logits, label)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
