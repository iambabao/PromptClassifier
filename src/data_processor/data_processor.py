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
from torch.utils.data import TensorDataset

from src.utils import read_file, read_json_lines

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, soft_prompt, hard_prompt, label):
        self.guid = guid
        self.context = context
        self.soft_prompt = soft_prompt
        self.hard_prompt = hard_prompt
        self.label = label

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
    def __init__(self, guid, input_ids, prompt_ids, attention_mask=None, token_type_ids=None, label=None):
        self.guid = guid
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer, max_seq_length, prompt_length):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        encoded = {"guid": example.guid}

        if prompt_length > 0:
            # Use mask token as placeholder for soft prompt
            input_text = (' '.join([tokenizer.mask_token] * prompt_length) + ' ' + example.hard_prompt, example.context)
        else:
            input_text = (example.hard_prompt, example.context)
        encoded.update(tokenizer.encode_plus(
            input_text,
            padding="max_length",
            truncation="longest_first",
            max_length=max_seq_length,
        ))
        # Each prompt has it's own ids
        encoded["prompt_ids"] = [
            _ for _ in range(example.soft_prompt * prompt_length, (example.soft_prompt + 1) * prompt_length)
        ]
        encoded["label"] = example.label

        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("prompt_ids: {}".format(encoded["prompt_ids"]))
            logger.info("attention_mask: {}".format(encoded["attention_mask"]))
            logger.info("label: {}".format(encoded["label"]))
            logger.info("input text: {}".format(input_text))

    return features


class DataProcessor:
    def __init__(
            self,
            model_type,
            model_name_or_path,
            max_seq_length,
            prompt_length,
            do_lower_case=False,
            data_dir="",
            cache_dir="cache",
            overwrite_cache=False
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.prompt_length = prompt_length

        self.data_dir = data_dir
        self.cache_dir = os.path.join(
            data_dir,
            "{}_{:02d}_{}".format(cache_dir, prompt_length, "uncased" if do_lower_case else "cased")
        )
        self.overwrite_cache = overwrite_cache

        os.makedirs(self.cache_dir, exist_ok=True)

    def load_and_cache_data(self, tokenizer, split, suffix=None):
        if suffix is not None:
            split = "{}_{}".format(split, suffix)

        cached_examples = os.path.join(self.cache_dir, "cached_example_{}".format(split))
        if os.path.exists(cached_examples) and not self.overwrite_cache:
            logger.info("Loading examples from cached file {}".format(cached_examples))
            examples = torch.load(cached_examples)
        else:
            examples = []
            for sample in tqdm(
                read_json_lines(os.path.join(self.data_dir, "data_{}.json".format(split))), desc="Loading Examples",
            ):
                sample['guid'] = len(examples)
                examples.append(InputExample(**sample))
            logger.info("Saving examples into cached file {}".format(cached_examples))
            torch.save(examples, cached_examples)

        cached_features = os.path.join(
            self.cache_dir,
            "cached_feature_{}_{}_{}".format(
                split,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                self.max_seq_length,
            ),
        )
        if os.path.exists(cached_features) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_features))
            features = torch.load(cached_features)
        else:
            features = convert_examples_to_features(examples, tokenizer, self.max_seq_length, self.prompt_length)
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
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids, all_prompt_ids, all_attention_mask, all_token_type_ids, all_label,
        )

        return dataset
