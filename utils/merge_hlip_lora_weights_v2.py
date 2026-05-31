import os
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer
from src.model.llm.hlip_qwen import VLMQwenForCausalLM


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="output/qwen3-0.6b",
        metadata={"help": "Path to the base LLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")

    # HLIP Vision encoder configuration
    vision_encoder_type: Optional[str] = field(
        default="singlescan_h2_token2744",
        metadata={"help": "HLIP vision encoder type used during training."}
    )
    proj_out_num: int = field(
        default=256, metadata={"help": "images patch number."}
    )
    vision_pretrained: bool = field(default=False)
    pretrain_vision_model: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained HLIP vision model weights, if used."}
    )
    freeze_vision_tower: bool = field(
        default=False, 
        metadata={"help": "Set to the same value as in training."}
    )
    
    model_with_lora: Optional[str] = field(
        default="./output/HLIP-VLM-train/model_with_lora.bin",
        metadata={"help": "Path to the saved model weights with LoRA adapters."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(
        default="./output/HLIP-VLM-merged",
        metadata={"help": "Directory to save the merged model."},
    )
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    cache_dir: Optional[str] = field(default=None)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
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


def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    print("=" * 20 + " 1. Tokenizer preparation " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    special_token = {"additional_special_tokens": ["<im_patch>"]}
    tokenizer.add_special_tokens(special_token)

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if "qwen" in model_args.model_type:
        if hasattr(tokenizer, 'eos_token_id'):
            tokenizer.pad_token = tokenizer.eos_token
            
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    print(f"Vocab size: {model_args.vocab_size}")

    print("=" * 20 + " 2. Model Reconstruction " + "=" * 20)
    model = VLMQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir
    )
    
    model.get_model().initialize_vision_modules(model_args=model_args)
    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)
    
    print("=" * 20 + " 3. Applying LoRA Config " + "=" * 20)
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
        print("Adding LoRA adapters to the model...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        raise ValueError("lora_enable must be set to True to merge weights.")

    print("=" * 20 + " 4. Loading Trained LoRA Weights " + "=" * 20)
    print(f"Loading weights from: {model_args.model_with_lora}")
    state_dict = torch.load(model_args.model_with_lora, map_location="cpu")
    
    # 打印加载的权重键名，用于调试
    print("Loaded state_dict keys (first 10):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}")
    if len(state_dict.keys()) > 10:
        print(f"  ... and {len(state_dict.keys()) - 10} more keys")
    
    model.load_state_dict(state_dict, strict=True)
    
    print("=" * 20 + " 5. Merging LoRA Weights " + "=" * 20)
    model = model.merge_and_unload()
    print("Weights merged successfully.")

    print("=" * 20 + " 6. Saving Merged Model " + "=" * 20)
    output_dir = training_args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving merged model to: {output_dir}")
    
    # 获取合并后的完整状态字典
    merged_state_dict = model.state_dict()
    
    # 打印保存的权重键名，用于调试
    print("Merged model state_dict keys (first 10):")
    for i, key in enumerate(list(merged_state_dict.keys())[:10]):
        print(f"  {key}")
    if len(merged_state_dict.keys()) > 10:
        print(f"  ... and {len(merged_state_dict.keys()) - 10} more keys")
    
    # 保存模型和tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 可选：单独保存vision_tower权重（如果需要的话）
    if hasattr(model.get_model(), 'vision_tower'):
        print("Saving vision tower weights separately...")
        vision_tower_weights = model.get_model().vision_tower.state_dict()
        torch.save(vision_tower_weights, os.path.join(output_dir, "vision_tower.bin"))
    
    # 验证保存的模型能否正确加载
    print("=" * 20 + " 7. Verification " + "=" * 20)
    print("Testing model loading...")
    try:
        # 尝试重新加载模型进行验证
        verification_model = VLMQwenForCausalLM.from_pretrained(output_dir)
        verification_model.get_model().initialize_vision_modules(model_args=model_args)
        print("✓ Model can be loaded successfully!")
        
        # 比较权重键名
        original_keys = set(merged_state_dict.keys())
        loaded_keys = set(verification_model.state_dict().keys())
        
        missing_keys = original_keys - loaded_keys
        unexpected_keys = loaded_keys - original_keys
        
        if missing_keys:
            print(f"Missing keys in loaded model: {list(missing_keys)[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys in loaded model: {list(unexpected_keys)[:5]}...")
        
        if not missing_keys and not unexpected_keys:
            print("✓ All keys match perfectly!")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")

    print("=" * 20 + " Finish " + "=" * 20)


if __name__ == "__main__":
    main()