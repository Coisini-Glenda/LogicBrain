#!/bin/bash
export METEOR_JAR=src/metrics/meteor/meteor-1.5.jar


OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29504 src/eval/eval_vqa_hlip.py \
    --model_name_or_path /data/huangxiaofei/Brian_CT/models/HLIP_QWEN3-8B_v2 \
    --data_root cap_data\
    --vqa_csv_filename cq500_qa.csv\
    --image_folder CQ500 \
    --max_length 512 \
    --do_sample \
    --output_dir /data/huangxiaofei/Brian_CT/eval_output/eval_CQ500