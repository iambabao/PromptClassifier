#!/usr/bin/env bash

set -e

SEED=2333
DATE=$(date +%m%d)

MODEL_TYPE="bert_prompt"
MODEL_NAME_OR_PATH="bert-base-cased"
MAX_SEQ_LENGTH=256
PROMPT_LENGTH=5

DATA_TYPE="example"
DATA_DIR="data/${DATA_TYPE}"
OUTPUT_DIR="checkpoints/${DATA_TYPE}/${DATE}"
CACHE_DIR="${HOME}/003_downloads/cache_transformers"
LOG_DIR="log/${DATA_TYPE}/${DATE}"

NUM_TRAIN_EPOCH=10
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
LEARNING_RATE=5e-5
LOGGING_STEPS=1000

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 23333 do_train.py \
--model_type ${MODEL_TYPE} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--prompt_length ${PROMPT_LENGTH} \
--data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} \
--cache_dir ${CACHE_DIR} \
--log_dir ${LOG_DIR} \
--do_train \
--do_eval \
--evaluate_during_training \
--overwrite_output_dir \
--num_train_epochs ${NUM_TRAIN_EPOCH} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--logging_steps ${LOGGING_STEPS} \
--seed ${SEED}
