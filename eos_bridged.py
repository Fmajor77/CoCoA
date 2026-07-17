import logging
import sys
import torch
from transformers import HfArgumentParser, TrainerCallback, TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

from src.model import MMEBModelStage1_5, create_block_attention_mask
from src.dataset import Stage1_5DatasetWithBridgeToken, Stage1_5Collator
from src.trainer import Stage1_5Trainer
from src.arguments import ModelArguments, DataArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def print_rank(msg):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


@dataclass
class Stage1_5TrainingArguments(TrainingArguments):
    mask_probability: float = field(default=0.7)
    
    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False
        print_rank(f"✓ remove_unused_columns set to: {self.remove_unused_columns}")


def load_processor(model_args):
    from transformers import AutoProcessor, AutoConfig
    from src.model_utils import get_backbone_name, QWEN2_VL, QWEN2_5_VL
    
    config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=config)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name, 
        trust_remote_code=True
    )
    
    return processor


def main():
    # Handle local_rank argument for distributed training
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, Stage1_5TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.remove_unused_columns = False
    model = MMEBModelStage1_5.build(model_args, training_args)
    processor = load_processor(model_args)
    
    train_dataset = Stage1_5DatasetWithBridgeToken(
        data_args, 
        model_args, 
        processor, 
        model.bridge_token_id
    )

    collator = Stage1_5Collator(
        data_args, 
        model_args, 
        processor, 
        model.bridge_token_id
    )
    trainer = Stage1_5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        mask_probability=training_args.mask_probability,
    )
    
    try:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)      
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()