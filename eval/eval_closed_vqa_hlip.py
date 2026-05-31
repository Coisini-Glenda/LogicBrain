import argparse
import csv
import os
import random
import json
import fcntl  # For file locking
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 核心改动：导入 ClosedVQADatset
from src.dataset.mllm_dataset import ClosedVQADataset_HLIP
from torch.cuda.amp import autocast

os.environ["HF_HUB_OFFLINE"] = "1"

class CSVWriter:
    """Thread-safe CSV writer for multi-process environment"""
    def __init__(self, filepath, rank=0):
        self.filepath = filepath
        self.rank = rank
        self.is_initialized = False
        
    def write_header_if_needed(self):
        """Write CSV header if file doesn't exist (only rank 0)"""
        if self.rank == 0 and not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(["patient_id", "study_id", "Question", "Answer", "Pred", "is_correct", "rank"])
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            self.is_initialized = True
    
    def append_results(self, results_batch):
        """Append batch of results to CSV file with file locking"""
        if not self.is_initialized:
            self.write_header_if_needed()
            self.is_initialized = True
            
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            for result in results_batch:
                writer.writerow([
                    result["patient_id"], result["study_id"], result["question"], 
                    result["answer"], result["pred"], result["is_correct"], self.rank
                ])
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="path/to/your/hlip_model_checkpoint"
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data_root", type=str, default="cap_data")
    # 核心改动：修改CSV文件名参数
    parser.add_argument("--closed_vqa_csv_filename", type=str, default="closed_qa_with_mode.csv")
    parser.add_argument("--image_folder", type=str, default="brain_origin_data_processed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/eval_vqa_closed/",
    )
    parser.add_argument("--proj_out_num", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--write_frequency", type=int, default=1, help="Write to CSV every N batches")
    return parser.parse_args(args)

def postprocess_text(text):
    # 简化后处理，仅需去除首尾空格并转为小写以方便比较
    return text.strip().lower()

def clean_text_for_csv(text):
    if isinstance(text, str):
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        text = text.replace('"', '""')
    return text

def setup_qwen_tokenizer(tokenizer):
    """Setup correct tokenizer configuration for Qwen models"""
    print("=== Setting up Qwen Tokenizer ===")
    print(f"Original pad_token: {tokenizer.pad_token}")
    print(f"Original pad_token_id: {getattr(tokenizer, 'pad_token_id', None)}")
    print(f"eos_token: {tokenizer.eos_token}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token_id to: {tokenizer.pad_token_id}")
    
    tokenizer.padding_side = "left"
    print(f"Padding side: {tokenizer.padding_side}")
    print("=================================")
    
    return tokenizer

def prepare_batch_inputs(questions, tokenizer, device, max_length):
    """Prepare batch inputs and create attention_mask"""
    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    return input_ids, attention_mask

def evaluate_batch_predictions(decoded_preds, decoded_labels):
    """Evaluate a batch of predictions and return accuracy results"""
    results = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_clean = postprocess_text(pred)
        label_clean = postprocess_text(label)
        
        # 简单字符串匹配判断正确性
        is_correct = 1 if pred_clean == label_clean else 0
        
        results.append({
            "is_correct": is_correct
        })
    
    return results

def main():
    seed_everything(42)
    args = parse_args()

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
   
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
   
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl')
   
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Tokenizer setup - 与代码1保持一致
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    
    tokenizer = setup_qwen_tokenizer(tokenizer)

    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # Dataset and DataLoader - 使用ClosedVQADataset
    test_dataset = ClosedVQADataset_HLIP(
        args, tokenizer=tokenizer, mode="test"
    )

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False, rank=rank, num_replicas=world_size
        )
    else:
        sampler = None

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
        shuffle=False, drop_last=False, sampler=sampler
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name_or_path.split("/")[-1]
    csv_output_path = os.path.join(args.output_dir, f"{model_name}_eval_closed_vqa_realtime.csv")
    
    csv_writer = CSVWriter(csv_output_path, rank=rank)
    
    if world_size > 1:
        dist.barrier()
        
    if rank != 0:
        time.sleep(1)
    
    csv_writer.write_header_if_needed()
    
    if world_size > 1:
        dist.barrier()

    actual_model = model.module if hasattr(model, 'module') else model

    if rank == 0:
        print("Starting Closed-ended VQA evaluation...")
        print(f"Batch size: {args.batch_size}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Write frequency: every {args.write_frequency} batches")
        print(f"Model dtype: {next(actual_model.parameters()).dtype}")
   
    start_time = time.time()
    processed_batches = 0
    batch_results_cache = []
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(test_dataloader, desc=f"Rank {rank}"):
        # 批量数据提取
        patient_ids = batch["patient_id"]
        study_ids = batch["study_id"]
        questions = batch["question"]
        choices = batch.get("choices", None)  # 可能包含选择项
        answers = batch["answer"] 
        images = batch["image"].to(device)
        
        batch_size = len(questions)
        
        # 如果有选择项，可以将其整合到问题中
        if choices is not None and choices[0]:  # 检查是否有选择项
            processed_questions = []
            for q, c in zip(questions, choices):
                if c and c.strip():
                    processed_questions.append(f"{q} {c}")
                else:
                    processed_questions.append(q)
            questions = processed_questions
        
        input_ids, attention_mask = prepare_batch_inputs(
            questions, tokenizer, device, args.max_length
        )

        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            try:
                generation = actual_model.generate(
                    images=images,
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=1.05,
                    length_penalty=1.0,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except Exception as e:
                print(f"Generation failed for batch: {e}")
                generation = input_ids
        
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
        
        # 提取回答部分
        processed_texts = []
        for gen_text, q in zip(generated_texts, questions):
            if q in gen_text:
                question_end = gen_text.find(q) + len(q)
                response = gen_text[question_end:].strip()
            else:
                response = gen_text.strip()
            
            response = response.replace(tokenizer.eos_token, "").strip()
            if not response:
                response = gen_text.strip()
            
            processed_texts.append(response)

        # 评估批量预测结果
        batch_metrics = evaluate_batch_predictions(processed_texts, answers)

        # 准备写入数据
        for i, metrics in enumerate(batch_metrics):
            result = {
                "patient_id": clean_text_for_csv(patient_ids[i]),
                "study_id": clean_text_for_csv(study_ids[i]),
                "question": clean_text_for_csv(questions[i]),
                "answer": clean_text_for_csv(answers[i]),
                "pred": clean_text_for_csv(processed_texts[i]),
                "is_correct": metrics["is_correct"]
            }
            batch_results_cache.append(result)
            
            # 统计准确率
            total_correct += metrics["is_correct"]
            total_samples += 1

        processed_batches += 1
        
        # 定期写入CSV
        if processed_batches % args.write_frequency == 0:
            csv_writer.append_results(batch_results_cache)
            batch_results_cache = [] 
            
            if rank == 0:
                elapsed = time.time() - start_time
                total_processed = processed_batches * args.batch_size
                speed = total_processed / elapsed
                current_accuracy = total_correct / total_samples if total_samples > 0 else 0
                
                print(f"\n Progress Update (Rank {rank}):")
                print(f"   Processed: {total_processed} samples in {processed_batches} batches")
                print(f"   Speed: {speed:.2f} samples/sec")
                print(f"   Current Accuracy: {current_accuracy:.4f} ({total_correct}/{total_samples})")
                print(f"  Written to: {csv_output_path}")

    # 写入剩余结果
    if batch_results_cache:
        csv_writer.append_results(batch_results_cache)

    if world_size > 1:
        dist.barrier()

    # 最终结果统计（仅rank 0）
    if rank == 0:
        # 读取最终CSV文件计算总体准确率
        all_results = []
        try:
            with open(csv_output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_results.append({
                        'is_correct': int(row['is_correct'])
                    })
        except Exception as e:
            print(f"Error reading final results: {e}")

        if all_results:
            final_total_correct = sum(res['is_correct'] for res in all_results)
            final_total_samples = len(all_results)
            final_accuracy = final_total_correct / final_total_samples if final_total_samples > 0 else 0

            print("\n" + "="*50)
            print("FINAL CLOSED VQA EVALUATION RESULTS")
            print("="*50)
            print(f"Total samples processed: {final_total_samples}")
            print(f"Total correct: {final_total_correct}")
            print(f"Final Accuracy: {final_accuracy:.4f}")
           
            # 保存最终分数
            score_file = os.path.join(args.output_dir, f"{model_name}_closed_vqa_final_scores.json")
            with open(score_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "final_scores": {
                        "accuracy": final_accuracy,
                        "total_correct": final_total_correct,
                        "total_samples": final_total_samples
                    },
                    "batch_size": args.batch_size
                }, f, indent=2)

        total_time = time.time() - start_time
        print(f"\n Total evaluation time: {total_time:.2f} seconds")
        print(f"Results saved to: {csv_output_path}")
        print(f"Final scores saved to: {score_file}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()