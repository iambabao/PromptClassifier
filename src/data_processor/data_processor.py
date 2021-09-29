# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import os
import copy
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import coo_matrix, vstack
from torch.utils.data import TensorDataset

from src.utils import read_file, read_json_lines

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, relation, relation_id, entities):
        self.guid = guid
        self.context = context
        self.relation = relation
        self.relation_id = relation_id
        self.entities = entities

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, guid, input_ids, prompt_ids, attention_mask=None, token_type_ids=None, length=None, labels=None):
        self.guid = guid
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.length = length[0]
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer, max_seq_length, hidden_prompt):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        encoded = {"guid": example.guid, "prompt_ids": example.relation_id}

        input_text = (example.relation, example.context) if not hidden_prompt else example.context
        encoded.update(tokenizer.encode_plus(
            input_text,
            truncation="longest_first",
            max_length=max_seq_length,
            return_length=True,
            return_offsets_mapping=True,
        ))
        tokenizer.pad(encoded, padding="max_length", max_length=max_seq_length)

        char2token = []
        offset = encoded["input_ids"].index(tokenizer.sep_token_id) if not hidden_prompt else 0
        for char_index in range(len(example.context)):
            for token_index, (start, end) in enumerate(encoded["offset_mapping"][offset:]):
                if char_index < start:
                    char2token.append(token_index - 1 + offset)
                    break
                elif start <= char_index < end:
                    char2token.append(token_index + offset)
                    break

        labels = [[0] * max_seq_length for _ in range(max_seq_length)]
        for start, end in example.entities:
            if start >= len(char2token) or end >= len(char2token):
                continue
            labels[char2token[start]][char2token[end - 1]] = 1
        encoded["labels"] = coo_matrix(labels).reshape(1, max_seq_length * max_seq_length)

        del encoded["offset_mapping"]
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("relation: {}".format(example.relation))
            for start, end in example.entities:
                logger.info("golden entity: {}".format(example.context[start:end]))
            for start in range(max_seq_length):
                for end in range(start, max_seq_length):
                    if labels[start][end] == 1:
                        logger.info("labeled entity: {}".format(tokenizer.decode(encoded["input_ids"][start:end + 1])))

    return features


class DataProcessor:
    def __init__(
            self,
            model_type,
            model_name_or_path,
            max_seq_length,
            hidden_prompt=False,
            data_dir="",
            overwrite_cache=False
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.hidden_prompt = hidden_prompt

        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, "cache_{}".format("discrete" if not hidden_prompt else "continuous"))
        self.overwrite_cache = overwrite_cache

        self.relation_types = [_.strip() for _ in read_file(os.path.join(data_dir, "schema.txt"))]
        self.id2relation = {k: v for k, v in enumerate(self.relation_types)}
        self.relation2id = {v: k for k, v in enumerate(self.relation_types)}

        os.makedirs(self.cache_dir, exist_ok=True)

    def process_data_file(self, filename):
        for entry in read_json_lines(filename):
            rel2entities = defaultdict(set)
            for triple in entry['triples']:
                rel2entities[triple['relation']].add((triple['head']['start'], triple['head']['end']))
                rel2entities[triple['relation']].add((triple['tail']['start'], triple['tail']['end']))
            for relation, entities in rel2entities.items():
                yield {
                    'context': entry['context'],
                    'relation': relation,
                    'relation_id': self.relation2id[relation],
                    'entities': entities,
                }

    def load_and_cache_data(self, tokenizer, role, suffix=None):
        if suffix is not None:
            role = "{}_{}".format(role, suffix)

        cached_examples = os.path.join(self.cache_dir, "cached_example_{}".format(role))
        if os.path.exists(cached_examples) and not self.overwrite_cache:
            logger.info("Loading examples from cached file {}".format(cached_examples))
            examples = torch.load(cached_examples)
        else:
            examples = []
            for sample in tqdm(
                self.process_data_file(os.path.join(self.data_dir, "data_{}.json".format(role))),
                desc="Loading Examples",
            ):
                sample['guid'] = len(examples)
                examples.append(InputExample(**sample))
            logger.info("Saving examples into cached file {}".format(cached_examples))
            torch.save(examples, cached_examples)

        cached_features = os.path.join(
            self.cache_dir,
            "cached_feature_{}_{}_{}".format(
                role,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                self.max_seq_length,
            ),
        )
        if os.path.exists(cached_features) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_features))
            features = torch.load(cached_features)
        else:
            features = convert_examples_to_features(examples, tokenizer, self.max_seq_length, self.hidden_prompt)
            logger.info("Saving features into cached file {}".format(cached_features))
            torch.save(features, cached_features)

        return examples, self._create_tensor_dataset(features)

    def _create_tensor_dataset(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_prompt_ids = torch.tensor([f.prompt_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        if self.model_type in ["bert", "xlnet", "albert"]:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([[0] * self.max_seq_length for _ in features], dtype=torch.long)
        all_length = torch.tensor([f.length for f in features], dtype=torch.long)
        all_labels = vstack([f.labels for f in features])
        all_labels = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([all_labels.row, all_labels.col]), dtype=torch.long),
            torch.tensor(all_labels.data, dtype=torch.long),
            size=all_labels.shape,
            dtype=torch.long,
        )

        dataset = TensorDataset(
            all_input_ids, all_prompt_ids, all_attention_mask, all_token_type_ids, all_length, all_labels
        )

        return dataset


def run_test():
    from transformers import AutoTokenizer
    from src.utils import init_logger

    init_logger(logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    processor = DataProcessor(
        'bert', 'bert-base-cased', max_seq_length=256, data_dir='../../data/NYT', overwrite_cache=True
    )
    processor.load_and_cache_data(tokenizer, 'test')


if __name__ == '__main__':
    run_test()
