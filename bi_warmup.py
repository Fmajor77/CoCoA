import logging
import sys
import torch
import wandb
from transformers import HfArgumentParser, TrainerCallback

from src.dataset import Stage1TextImageDataset
from src.collator import Stage1TrainCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModelStage1
from src.trainer import Stage1Trainer
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name

logger = logging.getLogger(__name__)

class AccuracyLoggingCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            if hasattr(model, '_fwd_debug'):
                model._fwd_debug = 0

def main():
    # Handle torch.distributed.launch arguments
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    if not hasattr(training_args, 'mask_probability'):
        training_args.mask_probability = 0.2

    if not hasattr(training_args, 'use_mae'):
        training_args.use_mae = True
    
    if not hasattr(training_args, 'mae_mask_ratio'):
        training_args.mae_mask_ratio = 0.3
    
    if not hasattr(training_args, 'mae_loss_weight'):
        training_args.mae_loss_weight = 0.5
    
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or \
           (not torch.distributed.is_initialized()):
            print_rank('Initializing wandb for Stage 1 MNTP+MAE training')
            wandb.init(
                project=training_args.project_name, 
                name=f"{training_args.run_name}_stage1_mntp_mae", 
                mode="online"
            )
    model = MMEBModelStage1.build(model_args, training_args)
    
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    
    if training_args.use_mae:
        print_rank(f'  - Mask ratio: {training_args.mae_mask_ratio}')
        print_rank(f'  - Loss weight: {training_args.mae_loss_weight}')
    
    print_rank("\nLoading processor...")
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)
    
    print_rank(f"\nLoading Stage 1 training dataset from {data_args.dataset_name}")
    train_dataset = Stage1TextImageDataset(data_args, model_args, processor)
    print_rank(f"Loaded {len(train_dataset)} training examples")
    
    collator = Stage1TrainCollator(
        data_args, 
        model_args, 
        processor,
        use_mae=training_args.use_mae,
        mae_mask_ratio=training_args.mae_mask_ratio
    )
    
    trainer = Stage1Trainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        mask_probability=training_args.mask_probability
    )
    trainer.add_callback(AccuracyLoggingCallback(log_every_n_steps=100))
    train_dataset.trainer = trainer
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()