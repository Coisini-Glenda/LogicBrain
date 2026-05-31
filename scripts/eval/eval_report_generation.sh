#!/bin/bash
export METEOR_JAR=src/metrics/meteor/meteor-1.5.jar

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29500 src/eval/eval_caption_hlip.py \
    --model_name_or_path /data/huangxiaofei/Brian_CT/models/HLIP_QWEN3-8B_v2 \
    --data_root cap_data \
    --cap_csv_filename report_with_mode.csv \
    --image_folder Brain_CT_Images \
    --max_length 1024 \
    --batch_size 1 \
    --do_sample \
    --output_dir /data/huangxiaofei/Brian_CT/eval_output/eval_HLIP_VLM_8b_Report_v2