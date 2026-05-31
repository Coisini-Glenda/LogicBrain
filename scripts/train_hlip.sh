#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5 --master_port=29501  src/train/train_hlip.py \
    --deepspeed scripts/zero2.json \
    --language_model_name_or_path output/bert-base-uncased \
    --wb_name HLIP \
    --loss_type "sigmoid" \
    --data_root Brain_CT_Images \
    --csv_data_path cap_data/report_with_mode.csv \
    --max_length 512 \
    --bf16 True \
    --output_dir ./output/HLIP_ceshi \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory False \
    --dataloader_num_workers 8