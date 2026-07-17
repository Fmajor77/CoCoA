import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

def print_rank(msg):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)

class Stage1TextImageDataset(Dataset):    
    def __init__(self, data_args, model_args, processor):
        self.data_args = data_args
        self.model_args = model_args
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        
        self.max_image_size = getattr(data_args, 'max_image_size', 336)
        self.items = self._load_items()
        
        print_rank(f"✓ Loaded {len(self.items)} items (Manual mode)")
    
    def _load_items(self) -> List[Dict]:
        dataset_dir = Path(self.data_args.dataset_name)
        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_dir}")      
        json_files = list(dataset_dir.glob("*.json"))       
        user_num_sample = getattr(self.data_args, 'num_sample_per_subset', None)  
        all_items = []      
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                
                valid_items = []
                for item in items:
                    if not isinstance(item, dict):
                        continue      
                    if 'image_path' not in item or 'caption' not in item:
                        if len(all_items) == 0 and len(valid_items) == 0:
                            print_rank(f"Sample fields: {list(item.keys())}")
                        continue
                    
                    img_path = item['image_path']
                    if not os.path.isabs(img_path):
                        base_dir = (self.data_args.image_dir 
                                if hasattr(self.data_args, 'image_dir') and self.data_args.image_dir 
                                else str(dataset_dir))
                        img_path = os.path.join(base_dir, img_path)
                        item['image_path'] = img_path
                    
                    valid_items.append(item)
                
                original_count = len(valid_items)
                all_items.extend(valid_items)
                
            except Exception as e:
                print_rank(f"  ✗ Error loading {json_file}: {e}")
        
        return all_items
    
    def _resize_image(self, image, max_dim=1344):
        if image is None:
            return None      
        resolution = getattr(self.data_args, 'image_resolution', None)
        
        if resolution == "high":
            image = image.resize((1344, 1344))
        elif resolution == "mid":
            image = image.resize((672, 672))
        elif resolution == "low":
            image = image.resize((336, 336))
        else:
            max_size = getattr(self.data_args, 'max_image_size', max_dim)
            cur_max_dim = max(image.size)
            if cur_max_dim > max_size:
                image = image.resize((max_size, max_size))
        
        return image
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item['image_path']      
        try:
            image = Image.open(image_path).convert('RGB')
            image = self._resize_image(image)
            if min(image.size) < 56:
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        except Exception as e:
            if idx < 10:
                print_rank(f"Warning: Failed to load {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))       
        caption = item['caption']
        text = f"<|vision_start|><|image_pad|><|vision_end|>{caption}"
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dim() > 0 and v.size(0) == 1:
                    result[k] = v.squeeze(0)
                else:
                    result[k] = v
            elif isinstance(v, list):
                if len(v) == 1:
                    item_v = v[0]
                    if isinstance(item_v, torch.Tensor):
                        while item_v.dim() > 2 and item_v.size(0) == 1:
                            item_v = item_v.squeeze(0)
                        if item_v.dim() == 2 and item_v.size(0) == 1:
                            item_v = item_v.squeeze(0)
                    result[k] = item_v
                else:
                    result[k] = v
            else:
                result[k] = v
        return result

class Stage1_5DatasetWithBridgeToken(Dataset):
    def __init__(
        self,
        data_args,
        model_args,
        processor,
        bridge_token_id: int,
    ):
        self.data_args = data_args
        self.model_args = model_args
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.bridge_token_id = bridge_token_id
        
        self.max_len = getattr(data_args, 'max_len', 512)
        self.items = self._load_items()
        
        print_rank(f"✓ Stage 1.5 Dataset: {len(self.items)} items")
        print_rank(f"  Bridge token ID: {bridge_token_id}")
    
    def _load_items(self) -> List[Dict]:
        dataset_dir = Path(self.data_args.dataset_name)
        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_dir}")
        
        json_files = list(dataset_dir.glob("*.json"))
        
        user_sample_config = getattr(self.data_args, 'num_sample_per_subset', None)
        
        default_sample_config = {
        }
        
        all_items = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                
                valid_items = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if 'qry' not in item or 'pos_text' not in item:
                        continue
                    
                    if 'qry_image_path' in item and item['qry_image_path']:
                        img_path = item['qry_image_path']
                        if not os.path.isabs(img_path):
                            base_dir = self.data_args.image_dir if hasattr(self.data_args, 'image_dir') and self.data_args.image_dir else str(dataset_dir)
                            img_path = os.path.join(base_dir, img_path)
                        item['qry_image_path'] = img_path
                    
                    valid_items.append(item)
                
                original_count = len(valid_items)
                file_name = json_file.stem
                
                num_sample = self._get_sample_size(
                    file_name=file_name,
                    original_count=original_count,
                    user_config=user_sample_config,
                    default_config=default_sample_config
                )
                
                if num_sample and len(valid_items) > num_sample:
                    valid_items = random.sample(valid_items, num_sample)
                    print_rank(f"  ✓ Loaded {len(valid_items)}/{original_count} from {json_file.name} (sampled)")
                else:
                    print_rank(f"  ✓ Loaded {len(valid_items)} from {json_file.name} (all)")
                
                all_items.extend(valid_items)
                
            except Exception as e:
                print_rank(f"  ✗ Error loading {json_file}: {e}")
        
        return all_items
    
    def _get_sample_size(
        self,
        file_name: str,
        original_count: int,
        user_config,
        default_config: Dict[str, int]
    ) -> Optional[int]:
        
        if isinstance(user_config, dict):
            if file_name in user_config:
                num = user_config[file_name]
                return num

            for key, num in user_config.items():
                if key in file_name or file_name in key:
                    return num
        
        elif isinstance(user_config, int):
            return user_config
        
        for key, num in default_config.items():
            if key in file_name or file_name in key:
                return num

    
    def _resize_image(self, image):
        if image is None:
            return None
        resolution = getattr(self.data_args, 'image_resolution', 'low')
        sizes = {'high': 1344, 'mid': 672, 'low': 448}
        size = sizes.get(resolution, 448)
        return image.resize((size, size))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        try:
            item = self.items[idx]
            image = None
            img_path = item.get('qry_image_path', '')
            if img_path and os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = self._resize_image(image)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    image = None
            
            qry_text = item.get('qry', '')
            qry_text = re.sub(r'<\|image_\d+\|>', '<|vision_start|><|image_pad|><|vision_end|>', qry_text)
            
            if image is not None and '<|vision_start|>' not in qry_text:
                qry_text = '<|vision_start|><|image_pad|><|vision_end|>' + qry_text
            
            pos_text = item.get('pos_text', '')
            
            if image is not None:
                block_a_inputs = self.processor(
                    text=[qry_text],
                    images=[image],
                    padding=False,
                    return_tensors="pt",
                )
            else:
                block_a_inputs = self.processor(
                    text=[qry_text],
                    padding=False,
                    return_tensors="pt",
                )
            
            block_b_inputs = self.tokenizer(
                pos_text,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            
            block_a_ids = block_a_inputs['input_ids'].squeeze(0).long()
            block_b_ids = block_b_inputs['input_ids'].squeeze(0).long()
            bridge_token = torch.tensor([self.bridge_token_id], dtype=torch.long)
            
            input_ids = torch.cat([block_a_ids, bridge_token, block_b_ids], dim=0)
            attention_mask = torch.ones_like(input_ids)
            
            if input_ids.size(0) > self.max_len:
                input_ids = input_ids[:self.max_len]
                attention_mask = attention_mask[:self.max_len]
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            
            if 'pixel_values' in block_a_inputs:
                pv = block_a_inputs['pixel_values']
                if isinstance(pv, torch.Tensor):
                    while pv.dim() > 2 and pv.size(0) == 1:
                        pv = pv.squeeze(0)
                result['pixel_values'] = pv
            
            if 'image_grid_thw' in block_a_inputs:
                igt = block_a_inputs['image_grid_thw']
                if isinstance(igt, torch.Tensor):
                    while igt.dim() > 1 and igt.size(0) == 1:
                        igt = igt.squeeze(0)
                    if igt.dim() == 2 and igt.size(0) == 1:
                        igt = igt.squeeze(0)
                elif isinstance(igt, list) and len(igt) == 1:
                    igt = igt[0]
                    if isinstance(igt, torch.Tensor):
                        while igt.dim() > 1 and igt.size(0) == 1:
                            igt = igt.squeeze(0)
                result['image_grid_thw'] = igt
            
            # if idx < 2:
            #     bridge_pos = (input_ids == self.bridge_token_id).nonzero(as_tuple=True)[0]
            #     print_rank(f"\n[Sample {idx}] BLOCK_A: {len(block_a_ids)}, BLOCK_B: {len(block_b_ids)}, Bridge pos: {bridge_pos.tolist()}")
            #     print_rank(f"[Sample {idx}] input_ids dtype: {result['input_ids'].dtype}, shape: {result['input_ids'].shape}")
            
            result['input_ids'] = result['input_ids'].contiguous().long()
            result['attention_mask'] = result['attention_mask'].contiguous().long()
            if 'pixel_values' in result and isinstance(result['pixel_values'], torch.Tensor):
                result['pixel_values'] = result['pixel_values'].contiguous()
            if 'image_grid_thw' in result and isinstance(result['image_grid_thw'], torch.Tensor):
                result['image_grid_thw'] = result['image_grid_thw'].contiguous().long()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return {
                'input_ids': torch.tensor([self.tokenizer.pad_token_id or 0], dtype=torch.long),
                'attention_mask': torch.tensor([1], dtype=torch.long),
            }
        
