#!/bin/bash
# export METEOR_JAR=src/metrics/meteor/meteor-1.5.jar  # 准确率评估不再需要METEOR

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29502 src/eval/eval_closed_vqa_hlip.py \
    --model_name_or_path /data/huangxiaofei/Brian_CT/models/HLIP_QWEN3-8B_v2 \
    --data_root cap_data \
    --closed_vqa_csv_filename closed_qa_with_mode.csv \
    --image_folder Brain_CT_Images \
    --max_length 512 \
    --batch_size 1\
    --num_workers 8\
    --do_sample \
    --output_dir /data/huangxiaofei/Brian_CT/eval_output/eval_HLIP_VLM_8b_Closed_VQA_v2