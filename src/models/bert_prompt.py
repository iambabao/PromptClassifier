# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/9/26
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/9/26
"""

import logging
import torch
import torch.nn as nn

from src.models.hf_bert import (
BertPreTrainedModel,
BertEncoder,
BertPooler,
BaseModelOutputWithPoolingAndCrossAttentions,
)
from src.models.utils import ce_loss

logger = logging.getLogger(__name__)


class BertEmbeddingsWithPrompt(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.prompt_length = config.prompt_length

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.prompt_embeddings is None:
            logger.info("Initialize prompt embeddings from scratch")
            self.prompt_embeddings = nn.Embedding(config.prompt_vocab_size, config.hidden_size)
        else:
            logger.info("Initialize prompt embeddings from {}".format(config.prompt_embeddings))
            initial = torch.tensor(torch.load(config.prompt_embeddings))
            self.prompt_embeddings = nn.Embedding.from_pretrained(initial, freeze=False)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
            self,
            input_ids=None,
            prompt_ids=None,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=None,
            prompts_embeds=None,
            past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # Construct embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if prompts_embeds is None:
            prompts_embeds = self.prompt_embeddings(prompt_ids)
        inputs_embeds = torch.cat(
            [inputs_embeds[:, :1, :], prompts_embeds, inputs_embeds[:, 1 + self.prompt_length:, :]], dim=1
        )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertModelWithPrompt(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.config = config

        self.embeddings = BertEmbeddingsWithPrompt(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_prompt_embeddings(self):
        return self.embeddings.prompt_embeddings

    def set_prompt_embeddings(self, value):
        self.embeddings.prompt_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            prompt_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            prompts_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            prompt_ids=prompt_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            prompts_embeds=prompts_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertClassifierWithPrompt(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prompt_length = config.prompt_length
        self.num_labels = config.num_labels

        self.bert = BertModelWithPrompt(config)
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
            prompt_ids=None,
            attention_mask=None,
            token_type_ids=None,
            length=None,
            labels=None,
    ):
        batch_size = input_ids.shape[0]
        max_seq_length = input_ids.shape[1]
        indexes = torch.arange(max_seq_length).expand(batch_size, max_seq_length).to(self.device)

        # mask for real tokens with shape (batch_size, max_seq_length)
        token_mask = torch.less(torch.greater(indexes, self.prompt_length + 1), length.unsqueeze(-1))
        # mask for valid spans with shape (batch_size, max_seq_length, max_seq_length)
        span_mask = torch.logical_and(
            token_mask.unsqueeze(-1).expand(-1, -1, max_seq_length),
            token_mask.unsqueeze(-2).expand(-1, max_seq_length, -1),
        ).triu()

        outputs = self.bert(
            input_ids,
            prompt_ids=prompt_ids,
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

    def set_gradient(self, with_gradient):
        for n, p in self.named_parameters():
            if not any(nd in n for nd in with_gradient):
                p.requires_grad = False

    def stop_gradient(self, no_gradient):
        for n, p in self.named_parameters():
            if any(nd in n for nd in no_gradient):
                p.requires_grad = False


def run_test():
    from src.utils import init_logger
    from transformers import AutoConfig

    init_logger(logging.INFO)
    config = AutoConfig.from_pretrained('bert-base-cased')
    config.prompt_length = 10
    config.prompt_vocab_size = 50
    model = BertClassifierWithPrompt.from_pretrained('bert-base-cased', config=config)

    for n, p in model.named_parameters():
        logger.info('name: {}, shape: {}, gradient: {}'.format(n, p.shape, p.requires_grad))


if __name__ == '__main__':
    run_test()
