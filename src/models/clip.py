# In[]
from torch import nn
from src.models.lora_sgp import LoRACLIPVisionTransformer
from src.models.lora_baseline import VanillaLoRACLIPVisionTransformer
from transformers import CLIPModel, CLIPProcessor
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_clip_model(args, train_mode="lora"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    if train_mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        return model, processor

    elif train_mode == "full":
        for n, p in model.named_parameters():
            if "vision_model.encoder.layers" in n and ("self_attn" in n or "mlp" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return model, processor

    elif train_mode == "lora":
        for p in model.parameters():
            p.requires_grad = False

        rank = args.lora_rank
        lora_type = getattr(args, 'lora_type', 'lora_sgp')
        
        if lora_type == 'lora_vanilla':
            # 普通 LoRA 基线（无 SGP/NSP 投影）
            alpha = getattr(args, 'lora_alpha', rank)
            dropout = getattr(args, 'lora_dropout', 0.0)
            model.vision_model = VanillaLoRACLIPVisionTransformer(
                model.vision_model,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout)
        
        elif lora_type == 'lora_nsp':
            # LoRA + NSP（硬投影）
            use_soft_projection = False
            model.vision_model = LoRACLIPVisionTransformer(
                model.vision_model,
                r=rank,
                use_soft_projection=use_soft_projection,
                nsp_eps=getattr(args, 'nsp_eps', 0.05),
                nsp_weight=getattr(args, 'nsp_weight', 0.02))
        
        elif lora_type == "lora_sgp":
            # LoRA + SGP（软投影）
            use_soft_projection = True
            model.vision_model = LoRACLIPVisionTransformer(
                model.vision_model,
                r=rank,
                weight_temp=getattr(args, 'weight_temp', 1.0),
                use_soft_projection=use_soft_projection,
                weight_kind=getattr(args, 'weight_kind', 'log1p'),
                weight_p=getattr(args, 'weight_p', 1.0))
        else:
            raise ValueError(f"Unsupported lora_type: {lora_type}")
        
        return model, processor

    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")


class CLIP_BaseNet(nn.Module):
    def __init__(self, args, train_mode="lora"):
        super(CLIP_BaseNet, self).__init__()
        self.train_mode = train_mode
        self.model, self.processor = get_clip_model(args, train_mode=train_mode)

    def forward(self, img, text):
        x = self.model.get_image_features(img)
        y = self.model.get_text_features(text)
        return x, y

    def encode_image(self, img):
        return self.model.get_image_features(img)

    def encode_text(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        return text_features

    @property
    def feature_dim(self):
        return self.model.config.projection_dim

# In[]
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", type=str, default="lora", choices=["frozen", "full", "lora"])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_type", type=str, default="lora_sgp", choices=["lora_sgp", "lora_nsp"])
    parser.add_argument("--nsp_eps", type=float, default=0.1)
    parser.add_argument("--nsp_weight", type=float, default=1.0)
    parser.add_argument("--weight_temp", type=float, default=0.1)
    parser.add_argument("--weight_kind", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--weight_p", type=float, default=2.0)
    args = parser.parse_args()

    model = CLIP_BaseNet(args, train_mode=args.train_mode)
    print(model)