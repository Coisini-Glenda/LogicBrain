import logging
import os
import atexit
import signal
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer

import wandb
from src.dataset.mllm_dataset import CapDataset_HLIP, TextDatasets_HLIP, VQADataset_HLIP, ClosedVQADataset, TherapyPrognosisDataset
from src.model.llm.hlip_qwen import VLMQwenForCausalLM
from src.train.trainer import MLLMTrainer

import torch.distributed as dist

def cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier() 
            dist.destroy_process_group()
            print("Successfully destroyed process group")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def signal_handler(signum, frame):

    print(f"Received signal {signum}, cleaning up...")
    cleanup()
    sys.exit(0)

def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args):
    if is_rank_zero():
        print(*args)


@dataclass
class ModelArguments:
    wb_name: Optional[str] = field(default="HLIP_VLM")
    model_name_or_path: Optional[str] = field(
        default="output/qwen3-0.6b",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    # HLIP Vision encoder configuration
    vision_encoder_type: Optional[str] = field(
        default="singlescan_h2_token2744",
        metadata={"help": "HLIP vision encoder type: singlescan_h2_token2744, multiscan_h2_token588, multiscan_h2_token1176, multiscan_h3_token1176"}
    )
    proj_out_num: int = field(
        default=256, metadata={"help": "images patch number."}
    )
    vision_pretrained: bool = field(default=False)
    pretrain_vision_model: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained HLIP vision model weights."}
    )
    freeze_vision_tower: bool = field(default=False)


@dataclass
class DataArguments:
    data_root: str = field(
        default="./data/", metadata={"help": "Root directory for all data."}
    )

    image_folder: str = field(
        default="Brain_CT_Images"
    )

    cap_csv_filename: str = field(
        default="ceshi_report_mode.csv"
    )

    vqa_csv_filename: str = field(
        default="qa_with_mode.csv"
    )

    closed_vqa_csv_filename: str = field(
        default="closed_qa_with_mode.csv"
    )

    therapy_prognosis_csv_filename: str = field(
        default="prognosis_results_with_mode.csv"
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # Training configuration
    bf16: bool = True
    output_dir: str = "./output/HLIP-VLM-train"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = [
        "vision_tower",
        "embed_tokens",
        "lm_head",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


@dataclass
class DataCollator:
    def __call__(self, batch: list) -> dict:
        images, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch]
            for key in ("image", "input_id", "label", "attention_mask")
        )

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        return return_dict


def main():

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)
    
    try:
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
        # Load tokenizer from the given path with specified configurations
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        # Define and add special tokens for image patches
        special_token = {"additional_special_tokens": ["<im_patch>"]}
        tokenizer.add_special_tokens(special_token)

        if tokenizer.unk_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        if "qwen" in model_args.model_type:
            if hasattr(tokenizer, 'eos_token_id'):
                tokenizer.pad_token = tokenizer.eos_token

        # Convert special tokens to token IDs
        model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
        model_args.vocab_size = len(tokenizer)
        rank0_print("vocab_size: ", model_args.vocab_size)

        rank0_print("=" * 20 + " Model preparation " + "=" * 20)
        # Initialize HLIP-VLM model
        model = VLMQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=training_args.cache_dir
        )

        model.config.use_cache = False

        if model_args.freeze_backbone:
            model.model.requires_grad_(False)

        model.enable_input_require_grads()
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Initialize HLIP vision modules
        model.get_model().initialize_vision_modules(model_args=model_args)

        # Set number of new tokens for vision
        model_args.num_new_tokens = 1
        model.initialize_vision_tokenizer(model_args, tokenizer)

        # Load pretrained MLLM weights if specified
        if model_args.pretrain_mllm:
            ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
            model.load_state_dict(ckpt, strict=True)
            rank0_print("load pretrained MLLM weights.")

        # Apply LoRA if enabled
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            rank0_print("Adding LoRA adapters only on LLM.")
            model = get_peft_model(model, lora_config)

            # Keep vision tower, embed_tokens, and lm_head trainable
            for n, p in model.named_parameters():
                if any(
                    [
                        x in n
                        for x in [
                            "vision_tower",
                            "embed_tokens",
                            "lm_head",
                        ]
                    ]
                ):
                    p.requires_grad = True

            model.print_trainable_parameters()

        rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
        data_args.max_length = training_args.model_max_length
        data_args.proj_out_num = model_args.proj_out_num

        train_dataset = TextDatasets_HLIP(data_args, tokenizer, mode="train")
        eval_dataset = TextDatasets_HLIP(data_args, tokenizer, mode="validation")
        data_collator = DataCollator()

        rank0_print("=" * 20 + " Training " + "=" * 20)
        trainer = MLLMTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Initialize wandb if rank 0
        wandb_initialized = False
        try:
            if is_rank_zero():
                wandb.login()
                wandb.init(project="HLIP_VLM", name=model_args.wb_name)
                wandb_initialized = True
        except Exception as e:
            rank0_print(f"Failed to initialize wandb: {e}")

        # Check for existing checkpoints and resume if available
        if os.path.exists(training_args.output_dir):
            checkpoints = sorted(
                [
                    d
                    for d in os.listdir(training_args.output_dir)
                    if d.startswith("checkpoint-")
                    and os.path.isdir(os.path.join(training_args.output_dir, d))
                ],
                key=lambda x: int(x.split("-")[-1]) if "-" in x else 0,
            )
            if checkpoints:
                last_checkpoint = checkpoints[-1]
                resume_ckpt = os.path.join(training_args.output_dir, last_checkpoint)
                rank0_print(f"Resuming from checkpoint: {resume_ckpt}")
                trainer.train(resume_from_checkpoint=resume_ckpt)
            else:
                trainer.train()
        else:
            trainer.train()

        trainer.save_state()
        model.config.use_cache = True

        rank0_print("=" * 20 + " Save model " + "=" * 20)
        if training_args.lora_enable:
            state_dict_with_lora = model.state_dict()
            torch.save(
                state_dict_with_lora,
                os.path.join(training_args.output_dir, "model_with_lora.bin"),
            )
        else:
            safe_save_model_for_hf_trainer(
                trainer=trainer, output_dir=training_args.output_dir
            )


        if wandb_initialized and is_rank_zero():
            try:
                wandb.finish()
            except Exception as e:
                rank0_print(f"Error finishing wandb: {e}")

        rank0_print("Training completed successfully!")
        
    except Exception as e:
        rank0_print(f"Error during training: {e}")

        if 'wandb_initialized' in locals() and wandb_initialized and is_rank_zero():
            try:
                wandb.finish()
            except:
                pass
        raise
    finally:
 
        cleanup()


if __name__ == "__main__":
    main()