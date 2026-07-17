import logging
import torch
import numpy as np

from src.utils import print_master

from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
# from src.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.vlm_backbone.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGenerationBidirectional, 
)

from src.vlm_backbone.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLForConditionalGenerationBidirectional, 
)

logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
MODEL2BACKBONE = {  # keys are from hf_config.model_type
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

vlm_image_tokens = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
}
backbone2model_bidirectional = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGenerationBidirectional,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGenerationBidirectional,
}

def load_processor(model_args):
    """
    Load processor based on VLM backbone.
    """
    print_master('Loading processor')
    model_name = model_args.processor_name if model_args.processor_name else model_args.model_name
    if model_args.model_backbone == PHI3V:
        from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            trust_remote_code=True,
        )
    elif model_args.model_backbone == QWEN2_VL:
        from src.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            image_processor=image_processor, tokenizer=tokenizer,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
    elif model_args.model_backbone == QWEN2_5_VL:
        from src.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        from transformers import AutoProcessor
        # image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name)
        # tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        # processor = Qwen2_5_VLProcessor.from_pretrained(model_name, image_processor=image_processor, tokenizer=tokenizer, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config):
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])

    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    if image_exists:
        # dummy image inputs based on the first valid data point
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
        # make the batch full tensors
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
        image_sizes = torch.cat(image_sizes, dim=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts = model_inputs['text']
    if ('image' in model_inputs):
        images = model_inputs['image']
    else:
        images = [None] * len(texts)
        
    image_exists = False
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs

def Qwen2_VL_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    
    for idx, (text, image) in enumerate(zip(texts, images)):
        if image is None:
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            
            raw_thw = inputs['image_grid_thw']
            
            if isinstance(raw_thw, torch.Tensor):
                raw_thw = raw_thw.cpu().numpy()
            elif not isinstance(raw_thw, np.ndarray):
                raw_thw = np.array(raw_thw)
            
            if raw_thw.ndim == 0:
                fixed_thw = np.array([[1, 18, 18]])
                if idx == 0:
                    print(f"[WARNING] Sample {idx}: 0-d image_grid_thw detected, using default")
            elif raw_thw.ndim == 1:
                if raw_thw.shape[0] == 3:
                    fixed_thw = raw_thw.reshape(1, 3)
                else:
                    fixed_thw = np.array([[1, 18, 18]])
                    if idx == 0:
                        print(f"[WARNING] Sample {idx}: Invalid 1-d shape {raw_thw.shape}, using default")
            elif raw_thw.ndim == 2:
                if raw_thw.shape[1] == 3:
                    fixed_thw = raw_thw
                else:
                    fixed_thw = np.array([[1, 18, 18]])
                    if idx == 0:
                        print(f"[WARNING] Sample {idx}: Invalid 2-d shape {raw_thw.shape}, using default")
            else:
                fixed_thw = raw_thw.squeeze()
                if fixed_thw.ndim == 1 and fixed_thw.shape[0] == 3:
                    fixed_thw = fixed_thw.reshape(1, 3)
                else:
                    fixed_thw = np.array([[1, 18, 18]])
                    if idx == 0:
                        print(f"[WARNING] Sample {idx}: Too many dims {raw_thw.ndim}, using default")
            
            image_grid_thw.append(fixed_thw)

    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': images,
    }
    
    if image_exists:
        valid_pixel_values = []
        valid_image_grid_thw = []
        
        for idx, (pv, thw) in enumerate(zip(pixel_values, image_grid_thw)):
            if pv is not None and thw is not None:
                if isinstance(pv, torch.Tensor):
                    valid_pixel_values.append(pv)
                else:
                    valid_pixel_values.append(torch.from_numpy(pv))
                
                if isinstance(thw, torch.Tensor):
                    thw_tensor = thw
                else:
                    thw_tensor = torch.from_numpy(thw)
                
                if thw_tensor.dim() == 1:
                    thw_tensor = thw_tensor.unsqueeze(0)
                
                if thw_tensor.dim() != 2 or thw_tensor.shape[1] != 3:
                    print(f"[CRITICAL] Sample {idx}: Final thw shape invalid {thw_tensor.shape}, creating fallback")
                    thw_tensor = torch.tensor([[1, 18, 18]], dtype=torch.long)
                
                valid_image_grid_thw.append(thw_tensor)
        
        if len(valid_pixel_values) > 0:
            inputs['pixel_values'] = torch.cat(valid_pixel_values, dim=0)
            
            concatenated_thw = torch.cat(valid_image_grid_thw, dim=0)
            
            if concatenated_thw.dim() != 2 or concatenated_thw.shape[1] != 3:
                print(f"[EMERGENCY] After cat: shape is {concatenated_thw.shape}, creating fallback")
                batch_size = len(valid_image_grid_thw)
                concatenated_thw = torch.tensor(
                    [[1, 18, 18]] * batch_size, 
                    dtype=torch.long
                )
            
            inputs['image_grid_thw'] = concatenated_thw
        else:
            inputs['pixel_values'] = None
            inputs['image_grid_thw'] = None
    else:
        inputs['pixel_values'] = None
        inputs['image_grid_thw'] = None

    return inputs

process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
}
