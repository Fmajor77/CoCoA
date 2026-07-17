from src.model_utils import LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL
from src.model_utils import get_backbone_name, backbone2model
from src.vlm_backbone.qwen2_5_vl.bidirectional_modeling_qwen2_5_vl import Qwen2_5_VLBiForMNTP
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, PeftModel
import torch 


def merge_lora_data_and_save(base_model_path, adapter_path, export_path, is_casual=False):
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=config)
    print(f'Loading backbone [{model_backbone}]')
    
    if model_backbone in {QWEN2_5_VL}:
        config._attn_implementation = "flash_attention_2"
        config.vision_config._attn_implementation = "flash_attention_2"
        
        if is_casual:
            base_model = backbone2model[model_backbone].from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                config=config
            )
        else:
            base_model = Qwen2_5_VLBiForMNTP.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                config=config
            )
    elif model_backbone in {LLAVA_NEXT, QWEN2_VL}:
        config._attn_implementation = "flash_attention_2"
        config.vision_config._attn_implementation = "flash_attention_2"
        base_model = backbone2model[model_backbone].from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            config=config
        )
    else:
        raise ValueError(f"Unsupported model backbone: {model_backbone}")

    lora_config = LoraConfig.from_pretrained(adapter_path)
    lora_model = PeftModel.from_pretrained(base_model, adapter_path, config=lora_config)
    merged_model = lora_model.merge_and_unload()
    
    merged_model.save_pretrained(
        save_directory=export_path,
        max_shard_size="5GB",
    )
    
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(export_path)


if __name__ == '__main__':
    base_model = ""
    adapter_path = ""
    save_path = ""
    
    merge_lora_data_and_save(base_model, adapter_path, save_path, is_casual=False)
