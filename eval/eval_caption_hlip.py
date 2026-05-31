import argparse
import csv
import os
import random
import json
import fcntl  # For file locking
import time
from collections import defaultdict

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.dataset.mllm_dataset import CapDataset_HLIP
from torch.cuda.amp import autocast

os.environ["HF_HUB_OFFLINE"] = "1"

# --- Load evaluation metrics ---
bleu_path = os.path.abspath("src/metrics/bleu")
bert_path = os.path.abspath("src/metrics/bertscore")
meteor_path = os.path.abspath("src/metrics/meteor")
rouge_path = os.path.abspath("src/metrics/rouge")

bleu = evaluate.load(bleu_path)
bertscore = evaluate.load(bert_path)
meteor = evaluate.load(meteor_path)
rouge = evaluate.load(rouge_path)

print("Evaluation metrics loaded successfully!")

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
                writer.writerow(["question", "target", "pred", "bleu", "rouge1", "meteor", "bert_f1", "rank"])
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
                    result["question"], result["target"], result["pred"],
                    result["bleu"], result["rouge1"], result["meteor"], 
                    result["bert_f1"], self.rank
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
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data_root", type=str, default="cap_data")
    parser.add_argument("--cap_csv_filename", type=str, default="report_with_mode.csv")
    parser.add_argument("--image_folder", type=str, default="brain_origin_data_processed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/eval_caption/",
    )
    parser.add_argument("--proj_out_num", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--write_frequency", type=int, default=1, help="Write to CSV every N batches")
    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def clean_text_for_csv(text):
    if isinstance(text, str):
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        text = text.replace('"', '""')
    return text

def setup_tokenizer(tokenizer):
    """Setup correct tokenizer configuration"""
    print("=== Setting up Tokenizer ===")
    print(f"Original pad_token: {tokenizer.pad_token}")
    print(f"Original pad_token_id: {getattr(tokenizer, 'pad_token_id', None)}")
    print(f"eos_token: {tokenizer.eos_token}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"? Set pad_token_id to: {tokenizer.pad_token_id}")
    
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
    """Evaluate a batch of predictions and return metrics"""
    results = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        result = {}
        
        # BLEU Score
        try:
            bleu_score = bleu.compute(predictions=[pred], references=[label], max_order=1)
            result["bleu"] = bleu_score["bleu"]
        except Exception:
            result["bleu"] = 0.0
            
        # ROUGE Score
        try:
            rouge_score = rouge.compute(predictions=[pred], references=[label], rouge_types=["rouge1"])
            result["rouge1"] = rouge_score["rouge1"]
        except Exception:
            result["rouge1"] = 0.0
            
        # METEOR Score
        try:
            meteor_score = meteor.compute(predictions=[pred], references=[label])
            result["meteor"] = meteor_score["meteor"]
        except Exception:
            result["meteor"] = 0.0
            
        # BERTScore
        try:
            bert_score = bertscore.compute(
                predictions=[pred], references=[label],
                lang="en", model_type="output/roberta-large", num_layers=17
            )
            result["bert_f1"] = bert_score["f1"][0] if bert_score["f1"] else 0.0
        except Exception:
            result["bert_f1"] = 0.0
            
        results.append(result)
    
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

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    
    tokenizer = setup_tokenizer(tokenizer)

    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # Dataset and DataLoader
    test_dataset = CapDataset_HLIP(
        args, tokenizer=tokenizer, mode="test"
    )

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False, rank=rank, num_replicas=world_size
        )
    else:
        sampler = None

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True,
        shuffle=False, drop_last=False, sampler=sampler
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name_or_path.split("/")[-1]
    csv_output_path = os.path.join(args.output_dir, f"{model_name}_eval_caption_realtime.csv")
    
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
        print("Starting Caption evaluation...")
        print(f"Batch size: {args.batch_size}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Write frequency: every {args.write_frequency} batches")
        print(f"Model dtype: {next(actual_model.parameters()).dtype}")
   
    start_time = time.time()
    processed_batches = 0
    batch_results_cache = []
    total_metrics = defaultdict(list)
    
    for batch in tqdm(test_dataloader, desc=f"Rank {rank}"):

        questions = batch["question"]
        answers = batch["answer"] 
        images = batch["image"].to(device)
        
        batch_size = len(questions)
        
        input_ids, attention_mask = prepare_batch_inputs(
            questions, tokenizer, device, args.max_length
        )

        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            try:
                # Enhanced generation call with more parameters for better quality
                generation = actual_model.generate(
                    images=images,
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=10,
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
                print(f"??  Generation failed for batch: {e}")
                generation = input_ids
        
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
        
        # Process generated texts to remove input questions
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

        decoded_preds, decoded_labels = postprocess_text(processed_texts, answers)
        

        if rank == 0:  
            print(f"\n{'='*60}")
            print(f"?? Batch {processed_batches + 1} Sample Generations:")
            print(f"{'='*60}")
            
            for i in range(len(questions)):
                print(f"\n--- Sample {i+1}/{len(questions)} ---")
                print(f"?? Question: {questions[i][2500:]}")
                print(f"? Ground Truth: {answers[i]}")
                print(f"?? Generated: {processed_texts[i]}")
                print(f"-----------------")
        
        batch_metrics = evaluate_batch_predictions(decoded_preds, decoded_labels)

        for i, metrics in enumerate(batch_metrics):
            result = {
                "question": clean_text_for_csv(questions[i]),
                "target": clean_text_for_csv(answers[i]),
                "pred": clean_text_for_csv(processed_texts[i]),
                "bleu": metrics["bleu"],
                "rouge1": metrics["rouge1"],
                "meteor": metrics["meteor"],
                "bert_f1": metrics["bert_f1"]
            }
            batch_results_cache.append(result)
            
            for metric_name, metric_value in metrics.items():
                total_metrics[metric_name].append(metric_value)

        processed_batches += 1
        
        # Write results periodically
        if processed_batches % args.write_frequency == 0:
            csv_writer.append_results(batch_results_cache)
            batch_results_cache = [] 
            
            if rank == 0:
                elapsed = time.time() - start_time
                total_processed = processed_batches * args.batch_size
                speed = total_processed / elapsed
                
                current_avg_scores = {}
                for metric_name, values in total_metrics.items():
                    current_avg_scores[metric_name] = sum(values) / len(values)
                
                print(f"\n?? Progress Update (Rank {rank}):")
                print(f"   Processed: {total_processed} samples in {processed_batches} batches")
                print(f"   Speed: {speed:.2f} samples/sec")
                print(f"   Current Avg - BLEU: {current_avg_scores.get('bleu', 0):.4f}, "
                      f"ROUGE1: {current_avg_scores.get('rouge1', 0):.4f}, "
                      f"METEOR: {current_avg_scores.get('meteor', 0):.4f}, "
                      f"BERT-F1: {current_avg_scores.get('bert_f1', 0):.4f}")
                print(f"   ?? Written to: {csv_output_path}")

    # Write remaining results
    if batch_results_cache:
        csv_writer.append_results(batch_results_cache)

    if world_size > 1:
        dist.barrier()

    # Final results calculation and summary (only rank 0)
    if rank == 0:
        # Read all results from CSV for final summary
        all_results = []
        try:
            with open(csv_output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_results.append({
                        'bleu': float(row['bleu']),
                        'rouge1': float(row['rouge1']),
                        'meteor': float(row['meteor']),
                        'bert_f1': float(row['bert_f1'])
                    })
        except Exception as e:
            print(f"Error reading final results: {e}")

        if all_results:
            metrics = ["bleu", "rouge1", "meteor", "bert_f1"]
            final_avg_scores = {
                metric: sum(res[metric] for res in all_results) / len(all_results) 
                for metric in metrics
            }

            print("\n" + "="*50)
            print("?? FINAL CAPTION EVALUATION RESULTS")
            print("="*50)
            print(f"Total samples processed: {len(all_results)}")
            for metric, avg_score in final_avg_scores.items():
                print(f"{metric.upper()}: {avg_score:.4f}")
           
            # Save final scores to JSON
            score_file = os.path.join(args.output_dir, f"{model_name}_caption_final_scores.json")
            with open(score_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "final_scores": final_avg_scores,
                    "total_samples": len(all_results),
                    "batch_size": args.batch_size
                }, f, indent=2)

        total_time = time.time() - start_time
        print(f"\n?? Total evaluation time: {total_time:.2f} seconds")
        print(f"Results saved to: {csv_output_path}")
        print(f"Final scores saved to: {score_file}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()