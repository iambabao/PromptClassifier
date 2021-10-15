# -*- coding: utf-8 -*-

"""
@Author             : huggingface
@Date               : 2020/7/26
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/11
"""

import argparse
import logging
import os
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoConfig, AutoTokenizer, set_seed
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup

from src.models import *
from src.data_processor import DataProcessor
from src.utils import init_logger, save_json, save_json_lines, generate_outputs, refine_outputs, compute_metrics

MODEL_MAPPING = {
    "bert": BertClassifier,
    "bert_prompt": BertClassifierWithPrompt,
}

logger = logging.getLogger(__name__)


def train(args, data_processor, model, tokenizer, role):
    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    _, train_dataset = data_processor.load_and_cache_data(tokenizer, role)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
    )
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    current_score, best_score = 0.0, 0.0
    set_seed(args.seed)  # Added here for reproductibility
    model.zero_grad()
    for _ in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[2].to(args.device),
                "length": batch[4].to(args.device),
                "labels": batch[-1].to_dense().to(args.device),
            }
            if args.prompt_length > 0:
                inputs["prompt_ids"] = batch[1].to(args.device)
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
            if args.model_type in ["bert", "xlnet", "albert"]:
                inputs["token_type_ids"] = batch[3].to(args.device)

            loss = model(**inputs)[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            description = "Global step: {:>6d}, Loss: {:>.4f}".format(global_step, loss.item())
            epoch_iterator.set_description(description)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        results = evaluate(args, data_processor, model, tokenizer, role="eval", prefix=str(global_step))
                        current_score, best_score = results["F1"], max(best_score, results["F1"])

                # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.save_best:
                        if current_score >= best_score:
                            logger.info("Saving model checkpoint to %s", args.output_dir)
                            os.makedirs(args.output_dir, exist_ok=True)
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(args.output_dir)
                            tokenizer.save_pretrained(args.output_dir)
                            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                    else:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        os.makedirs(output_dir, exist_ok=True)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            break

    if not args.save_best:
        logger.info("Saving model checkpoint to %s", args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    return global_step, tr_loss / global_step


def evaluate(args, data_processor, model, tokenizer, role, prefix=""):
    if prefix == "":
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    examples, dataset = data_processor.load_and_cache_data(tokenizer, role)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("Num examples = %d", len(dataset))
    logger.info("Batch size = %d", args.eval_batch_size)

    eval_outputs = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[2].to(args.device),
                "length": batch[4].to(args.device),
                "labels": batch[-1].to_dense().to(args.device),
            }
            if args.prompt_length > 0:
                inputs["prompt_ids"] = batch[1].to(args.device)
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
            if args.model_type in ["bert", "xlnet", "albert"]:
                inputs["token_type_ids"] = batch[3].to(args.device)

            logits = model(**inputs)[1]

            predicted = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            eval_outputs.extend(generate_outputs(
                predicted,
                inputs["labels"].detach().cpu().numpy(),
                inputs["input_ids"].detach().cpu().numpy(),
                tokenizer,
            ))
    eval_outputs = refine_outputs(examples, eval_outputs)
    eval_outputs_file = os.path.join(output_dir, "{}_outputs.json".format(role))
    save_json_lines(eval_outputs, eval_outputs_file)

    eval_results = compute_metrics(eval_outputs)
    eval_results_file = os.path.join(output_dir, "{}_results.txt".format(role))
    with open(eval_results_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in eval_results.keys():
            logger.info("%s = %s", key, str(eval_results[key]))
            writer.write("%s = %s\n" % (key, str(eval_results[key])))

    return eval_results


def main():
    parser = argparse.ArgumentParser()

    # Model hyper parameters
    parser.add_argument("--model_type", required=True, type=str, help="Model type")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument("--prompt_length", default=0, type=int, help="The length of prompt.")
    parser.add_argument("--prompt_embeddings", default=None, type=str, help="Path to the initial embeddings of prompt.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    # Directory parameters
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="The log directory where the running details will be written.",
    )

    # Training parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the eval set.")
    parser.add_argument("--save_best", action="store_true", help="Whether to save the best model .")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--per_device_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )

    # Other parameters
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    # Setup output dir
    if args.do_train:
        args.output_dir = os.path.join(
                args.output_dir,
                "{}_{}_{}_{:02d}_{:.1e}".format(
                    args.model_type,
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                    args.max_seq_length,
                    args.prompt_length,
                    args.learning_rate,
                ),
            )
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # Setup log dir
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_file = os.path.join(
            args.log_dir,
            "{}_{}_{}_{:02d}_{:.1e}.txt".format(
                args.model_type,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_length,
                args.prompt_length,
                args.learning_rate,
            ),
        )
    else:
        args.log_file = None
    init_logger(logging.INFO, args.log_file)
    logger.warning(args)

    # Set seed
    set_seed(args.seed)

    # Load config, tokenizer and pretrained model
    data_processor = DataProcessor(
        args.model_type,
        args.model_name_or_path,
        args.max_seq_length,
        args.prompt_length,
        data_dir=args.data_dir,
        overwrite_cache=args.overwrite_cache,
    )
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=2,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        use_fast=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_class = MODEL_MAPPING[args.model_type]
    # Add custom params for model
    config.prompt_length = args.prompt_length
    config.prompt_vocab_size = args.prompt_length * len(data_processor.relation_types)
    config.prompt_embeddings = args.prompt_embeddings
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # TODO: Only update new layers
    # model.set_gradient(with_gradient=['prompt_embeddings', 'start_layer', 'end_layer', 'output_layer'])
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    logger.info("Training/evaluation config %s", config)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, data_processor, model, tokenizer, role="train")
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            try: int(global_step)
            except ValueError: global_step = ""

            # Reload the model
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, data_processor, model, tokenizer, role="test", prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(json.dumps(results, ensure_ascii=False, indent=4)))
    save_json(results, os.path.join(args.output_dir, "all_results.json"))


if __name__ == "__main__":
    main()
