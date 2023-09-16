#!/bin/bash

git clone https://github.com/FlagOpen/FlagEmbedding.git


#train base
torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ./finetuned_bge \
--model_name_or_path BAAI/bge-base-zh \
--train_data /kaggle/input/cail23/train_bge_data.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 4 \
--negatives_cross_device

