from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from .encoder.hlip_visual_encoder import (
    vit_base_singlescan_h2_token2744,
    vit_base_multiscan_h2_token588,
    vit_base_multiscan_h2_token1176,
    vit_base_multiscan_h3_token1176
)
class PatchProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_up = nn.Linear(768, 4096)  # 升维
        self.pool = nn.AdaptiveAvgPool1d(256) # 降采样patch数

    def forward(self, x):
        # x: (batch_size, 2048, 768)
        x = self.linear_up(x)  # (batch_size, 2048, 1048)
        x = x.permute(0, 2, 1)  # (batch_size, 1048, 2048)，方便池化
        x = self.pool(x)        # (batch_size, 1048, 256)
        x = x.permute(0, 2, 1)  # (batch_size, 256, 1048)
        return x

class HLIPVLMMetaModel:
    def __init__(self, config):
        super(HLIPVLMMetaModel, self).__init__(config)
        
        if hasattr(config, "vision_encoder_type"):
            self.vision_tower = self._build_hlip_vision_tower(config)
        
        self.mm_projector = PatchProjector()

    def _build_hlip_vision_tower(self, config):
        """Build HLIP vision tower based on config"""
        vision_encoder_type = getattr(config, 'vision_encoder_type', 'singlescan_h2_token2744')
        vision_pretrained = getattr(config, 'vision_pretrained', True)
        
        if vision_encoder_type == "singlescan_h2_token2744":
            return vit_base_singlescan_h2_token2744(pretrained=vision_pretrained)
        elif vision_encoder_type == "multiscan_h2_token588":
            return vit_base_multiscan_h2_token588(pretrained=vision_pretrained)
        elif vision_encoder_type == "multiscan_h2_token1176":
            return vit_base_multiscan_h2_token1176(pretrained=vision_pretrained)
        elif vision_encoder_type == "multiscan_h3_token1176":
            return vit_base_multiscan_h3_token1176(pretrained=vision_pretrained)
        else:
            raise ValueError(f"Unsupported vision encoder type: {vision_encoder_type}")

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args):
        # Set basic vision config parameters
        self.config.vision_encoder_type = getattr(model_args, 'vision_encoder_type', 'singlescan_h2_token2744')
        self.config.vision_pretrained = getattr(model_args, 'vision_pretrained', True)

        # Build vision tower if not exists
        if self.get_vision_tower() is None:
            self.vision_tower = self._build_hlip_vision_tower(self.config)
            
            # Set freeze/unfreeze based on model_args
            freeze_vision = getattr(model_args, 'freeze_vision_tower', False)
            self.vision_tower.requires_grad_(not freeze_vision)

        # Load pretrained vision model if specified
        if hasattr(model_args, 'pretrain_vision_model') and model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(
                model_args.pretrain_vision_model, map_location="cpu"
            )
            self.vision_tower.load_state_dict(vision_model_weights, strict=False)
        
        # 初始化 mm_projector（如果还没有的话）
        if not hasattr(self, 'mm_projector') or self.mm_projector is None:
            self.mm_projector = PatchProjector()


class HLIPVLMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """Encode images using HLIP vision tower"""
        vision_tower = self.get_model().get_vision_tower()
        image_features = vision_tower.forward_features(images) 
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        else:
            image_features = self.encode_images(images)   # image features_1: torch.Size([1, 1, 768])
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (
                    inputs_embeds[:, :1, :],  # First token (e.g., BOS)
                    image_features,           # Image features
                    inputs_embeds[:, (image_features.shape[1] + 1):, :],  # Remaining tokens
                ),
                dim=1,
            )
            
        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """Initialize vision-related tokenizer components"""
        num_new_tokens = getattr(model_args, 'num_new_tokens', 0)

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
