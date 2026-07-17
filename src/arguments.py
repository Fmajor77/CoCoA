from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_backbone: str = field(
        default=None,
        metadata={"help": "backbone name"}
    )
    model_type: str = field(
        default=None, metadata={"help": "lavis model type"}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint for resuming training or loading pre-trained model"}
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained processor name or path if different from model_name"}
    )
    pooling: str = field(
        default='last',
        metadata={"help": "Pooling method: 'last', 'mean', 'cls'"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the embeddings"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "Temperature for contrastive loss"}
    )
    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: str = field(
        default="gate_proj,fc1,k_proj,o_proj,q_proj,qkv,up_proj,v_proj,fc2,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    stage2_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to Stage 1 checkpoint (required for Stage 2 LoRA-only checkpoints)"}
    )
    num_crops: int = field(
        default=16,
        metadata={"help": "number of crops used in image encoder"}
    )
    
    use_eos_contrastive: bool = field(
        default=False,
        metadata={
            "help": "Whether to use EOS token contrastive learning (from Stage 1.5). "
                    "This will compute contrastive loss on both pooled representations "
                    "and EOS token representations."
        }
    )
    eos_weight: float = field(
        default=0.5,
        metadata={
            "help": "Weight for EOS contrastive loss. "
                    "Total loss = (1 - eos_weight) * pooled_loss + eos_weight * eos_loss. "
                    "Default: 0.5 (equal weighting)"
        }
    )
    eos_token_id: int = field(
        default=151645,
        metadata={
            "help": "Token ID for the EOS/bridge token used in Stage 1.5. "
        }
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_split: Optional[str] = field(
        default='train',
        metadata={"help": "Dataset split to use for evaluation"}
    )
    split_name: Optional[str] = field(
        default='original',
        metadata={"help": "Split name for the dataset"}
    )
    subset_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of subset names to use"}
    )
    num_sample_per_subset: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to use per subset"}
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing images"}
    )
    image_resolution: Optional[str] = field(
        default='low',
        metadata={"help": "Image resolution: 'low', 'mid', or 'high'"}
    )
    max_len: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    encode_output_path: str = field(
        default=None, metadata={"help": "encode output path"}
    )
    padding: Optional[str] = field(
        default=None,
        metadata={"help": "Padding strategy"}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    """
    Training arguments specific to our setup
    """
    # Gradient cache specific
    grad_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient cache"}
    )
    gc_q_chunk_size: int = field(
        default=4,
        metadata={"help": "Chunk size for query in gradient cache"}
    )
    gc_p_chunk_size: int = field(
        default=4,
        metadata={"help": "Chunk size for passage in gradient cache"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "directory for saving trained models"}
    )
    
    # Logging
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project name for wandb"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Run name for wandb"}
    )
    
    def __post_init__(self):
        super().__post_init__()
        # Ensure remove_unused_columns is False for our custom datasets
        self.remove_unused_columns = False

@dataclass
class Stage1_5TrainingArguments(TrainingArguments):
    mask_probability: float = field(default=0.7)
    patch_mask_probability: float = field(
        default=1.0,
        metadata={"help": "Probability of masking image patches in BLOCK_A"}
    )
    patch_mask_strategy: str = field(
        default="noise",
        metadata={"help": "Strategy for patch masking: 'zero', 'noise', 'mean'"}
    )
    save_patch_visualization: bool = field(
        default=True, 
        metadata={"help": "Save masked patch visualization for first batch"}
    )
    patch_vis_save_dir: str = field(
        default="./visualizations/masked_patches",
        metadata={"help": "Directory to save patch visualizations"}
    )
