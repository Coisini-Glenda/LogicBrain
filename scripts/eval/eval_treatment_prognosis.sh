#!/bin/bash
export METEOR_JAR=src/metrics/meteor/meteor-1.5.jar

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29504 src/eval/eval_cq500_tp.py \
    --model_name_or_path /data/huangxiaofei/Brian_CT/models/HLIP_QWEN3-8B \
    --data_root cap_data \
    --therapy_prognosis_csv_filename cq500_qa.csv \
    --image_folder CQ500 \
    --max_length 1280 \
    --max_new_tokens 1280\
    --batch_size 16\
    --do_sample \
    --output_dir /data/huangxiaofei/Brian_CT/eval_output/eval_tp_cq500_ceshi