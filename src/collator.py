from itertools import repeat
from torch.jit import isinstance

import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch
import numpy as np

from src.model_utils import LLAVA_NEXT, QWEN2_VL, PHI3V

logger = logging.getLogger(__name__)

PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

def process_vlm_inputs(model_inputs: dict, processor, backbone_name, max_length=None, insert_eos=False, eos_token_id=151643):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False

    for text, image in zip(texts, images):
        if image is None:
            if backbone_name == QWEN2_VL:
                inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
                input_id = inputs["input_ids"].squeeze().tolist()
                if isinstance(input_id, int):
                    input_id = [input_id]
                
                if insert_eos:
                    input_id = input_id + [eos_token_id]
                
                input_ids.append(input_id)
                pixel_values.append(None)
                image_sizes.append(None)
                image_grid_thw.append(None)
            elif backbone_name == LLAVA_NEXT:
                inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
                input_id = inputs["input_ids"].squeeze().tolist()
                if isinstance(input_id, int):
                    input_id = [input_id]
                
                if insert_eos:
                    input_id = input_id + [eos_token_id]
                
                input_ids.append(input_id)
                pixel_values.append(None)
                image_sizes.append(None)
                image_grid_thw.append(None)
            elif backbone_name == PHI3V:
                inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
                input_id = inputs["input_ids"].squeeze().tolist()
                if isinstance(input_id, int):
                    input_id = [input_id]
                
                if insert_eos:
                    input_id = input_id + [eos_token_id]
                
                input_ids.append(input_id)
                pixel_values.append(None)
                image_sizes.append(None)
                image_grid_thw.append(None)
        else:
            image_exists = True
            if backbone_name == QWEN2_VL:
                inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
                input_id_list = inputs["input_ids"].squeeze().tolist()
                
                if insert_eos:
                    input_id_list = input_id_list + [eos_token_id]
                
                input_ids.append(input_id_list)
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
            elif backbone_name == LLAVA_NEXT:
                inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
                input_id_list = inputs["input_ids"].squeeze().tolist()
                
                if insert_eos:
                    input_id_list = input_id_list + [eos_token_id]
                
                input_ids.append(input_id_list)
                pixel_values.append(inputs['pixel_values'])
                if 'image_sizes' in inputs:
                    image_sizes.append(inputs['image_sizes'])
                image_grid_thw.append(None)
            elif backbone_name == PHI3V:
                inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
                input_id_list = inputs["input_ids"].squeeze().tolist()
                
                if insert_eos:
                    input_id_list = input_id_list + [eos_token_id]
                
                input_ids.append(input_id_list)
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(None)

    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': images,
    }
    
    if image_exists:
        if backbone_name == QWEN2_VL:
            pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
            pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
            pixel_values = torch.stack(pixel_values, dim=0)
            
            inputs['pixel_values'] = pixel_values
            inputs['image_grid_thw'] = image_grid_thw
            
        elif backbone_name == LLAVA_NEXT:
            pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
            image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
            pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
            image_sizes = torch.cat(image_sizes, dim=0)
            
            inputs['pixel_values'] = pixel_values
            inputs['image_sizes'] = image_sizes
            
        elif backbone_name == PHI3V:
            pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
            pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            inputs['pixel_values'] = pixel_values
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        if backbone_name == QWEN2_VL:
            inputs['image_grid_thw'] = [None] * input_ids.shape[0]
        elif backbone_name == LLAVA_NEXT:
            inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def create_process_fn_with_eos(processor, backbone_name, max_length, insert_eos=False, eos_token_id):
    def process_fn(model_inputs: dict):
        return process_vlm_inputs(
            model_inputs, 
            processor=processor, 
            backbone_name=backbone_name,
            max_length=max_length,
            insert_eos=insert_eos,
            eos_token_id=eos_token_id
        )
    return process_fn
    
def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs


def split_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    # for pixel_values and image_sizes, need to split based on the position of images
    input_ids = arg_val["input_ids"]
    positions = torch.nonzero((input_ids < 0) & (input_ids > -PHI_IMAGE_TOKEN_MAX_INPUT_ID), as_tuple=True)
    row_contain_image = torch.unique(positions[0])
    num_chunks = len(chunked_tensors[0])
    chunk_image_count = []
    for chunk_idx in range(num_chunks):
        chunk_image_count.append(torch.sum(
            (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
    if "pixel_values" in keys:
        pixel_values = arg_val["pixel_values"]
        image_sizes = arg_val["image_sizes"]
        chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        if "pixel_values" in keys and tt[2].numel() == 0:
            chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
        else:
            chunked_arg_val.append(dict(zip(kk, tt)))

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


@dataclass
class TrainTextImageDataCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __post_init__(self):
        self.insert_eos = False
        if hasattr(self.model_args, 'use_eos_contrastive') and self.model_args.use_eos_contrastive:
            self.insert_eos = True
            self.eos_token_id = getattr(self.model_args, 'eos_token_id', 151643)
            logger.info(f"✓ Auto-insert EOS enabled in collator: token_id={self.eos_token_id}")
            logger.info(f"  EOS will be inserted in process_vlm_inputs during gradient cache")

    def __call__(self, examples):
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        texts, images = [], []
        for example in examples:
            if example is None or not example:
                text, image = '  ', None
            else:
                text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs


@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __post_init__(self):
        self.insert_eos = False
        self.eos_token_id = ""
        
        if hasattr(self.model_args, 'use_eos_contrastive') and self.model_args.use_eos_contrastive:
            self.insert_eos = True
            if hasattr(self.model_args, 'eos_token_id'):
                self.eos_token_id = self.model_args.eos_token_id
            logger.info(f"✓ EvalCollator: Auto-insert EOS enabled (token_id={self.eos_token_id})")

    def __call__(self, examples):
        examples = {'text': [e[0] for e in examples], 'image': [e[1] for e in examples]}
        
        inputs = process_vlm_inputs(
            model_inputs=examples,
            processor=self.processor,
            backbone_name=self.model_args.model_backbone,
            max_length=self.data_args.max_len,
            insert_eos=self.insert_eos,
            eos_token_id=self.eos_token_id
        )
        
        return inputs