import json
import sys
import os
import torch
import torch.distributed as dist
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import HfArgumentParser, AutoProcessor, AutoConfig

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from evaluation.eval_utils import get_pred
from src.utils import print_rank
from src.model_utils import get_backbone_name


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            r = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(r)
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if data_args.insert_eos and data_args.eos_token_id is not None:
        model_args.eos_token_id = data_args.eos_token_id
        if is_main_process:
            print(f"[INFO] insert_eos=True, eos_token_id={data_args.eos_token_id}")
    
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = training_args.device
    
    if is_main_process:
        os.makedirs(data_args.encode_output_path, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()

    if is_main_process:
        print("=" * 80)
        print("Evaluation Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Dataset: {data_args.dataset_name}")
        print(f"  Subsets: {data_args.subset_name}")
        print("=" * 80)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    
    if is_main_process:
        print(f'\nLoading model ({model_backbone})...')
    
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(device, dtype=torch.bfloat16)
    
    if is_main_process:
        print("✓ Model loaded\n")

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
        insert_eos=data_args.insert_eos,
        eos_token_id=data_args.eos_token_id,
    )

    is_local = os.path.exists(data_args.dataset_name)

    for idx, subset in enumerate(data_args.subset_name):
        if is_main_process:
            print("=" * 80)
            print(f"[{idx+1}/{len(data_args.subset_name)}] Processing {subset}")
            print("=" * 80)
        
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        skip = False
        
        if is_main_process:
            if os.path.exists(score_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    print(f"✓ Found result: {score_dict['acc']:.4f}")
                    skip = True
                except:
                    pass
        
        if world_size > 1:
            skip_tensor = torch.tensor([skip], dtype=torch.bool, device=device)
            dist.broadcast(skip_tensor, src=0)
            skip = skip_tensor.item()
        
        if skip:
            continue
        
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        
        already_encoded = False
        if is_main_process:
            already_encoded = os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path)
        
        if world_size > 1:
            encoded_tensor = torch.tensor([already_encoded], dtype=torch.bool, device=device)
            dist.broadcast(encoded_tensor, src=0)
            already_encoded = encoded_tensor.item()
        
        if not already_encoded:
            from copy import deepcopy
            eval_data_args = deepcopy(data_args)
            
            if is_local:
                eval_data_args.dataset_name = os.path.join(data_args.dataset_name, subset)
                eval_subset = None
            else:
                eval_data_args.dataset_name = data_args.dataset_name
                eval_subset = subset

            try:
                eval_qry_dataset = EvalDataset(
                    data_args=eval_data_args,
                    model_args=model_args,
                    subset=eval_subset,
                    text_field="qry_text",
                    img_path_field="qry_img_path",
                )
                
                eval_tgt_dataset = EvalDataset(
                    data_args=eval_data_args,
                    model_args=model_args,
                    subset=eval_subset,
                    text_field="tgt_text",
                    img_path_field="tgt_img_path",
                )
                
                if is_main_process:
                    print(f"  Queries: {len(eval_qry_dataset)}, Targets: {len(eval_tgt_dataset)}")
                
            except Exception as e:
                if is_main_process:
                    print(f" Failed to load {subset}: {e}")
                continue

            if world_size > 1:
                from torch.utils.data.distributed import DistributedSampler
                
                class NoPaddingSampler(DistributedSampler):
                    def __iter__(self):
                        indices = list(range(len(self.dataset)))
                        indices = indices[self.rank:len(indices):self.num_replicas]
                        return iter(indices)
                
                qry_sampler = NoPaddingSampler(eval_qry_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                tgt_sampler = NoPaddingSampler(eval_tgt_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                
                eval_qry_loader = DataLoader(
                    eval_qry_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    sampler=qry_sampler,
                    collate_fn=eval_collator,
                    num_workers=training_args.dataloader_num_workers,
                )
                eval_tgt_loader = DataLoader(
                    eval_tgt_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    sampler=tgt_sampler,
                    collate_fn=eval_collator,
                    num_workers=training_args.dataloader_num_workers,
                )
            else:
                eval_qry_loader = DataLoader(
                    eval_qry_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    collate_fn=eval_collator,
                    shuffle=False,
                    num_workers=training_args.dataloader_num_workers,
                )
                eval_tgt_loader = DataLoader(
                    eval_tgt_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    collate_fn=eval_collator,
                    shuffle=False,
                    num_workers=training_args.dataloader_num_workers,
                )

            if is_main_process:
                print(f"  Encoding queries...")
            
            if world_size > 1:
                local_indices = list(qry_sampler)
            else:
                local_indices = list(range(len(eval_qry_dataset)))
            
            encoded_tensor = []
            encoded_paired_data = []
            idx_pointer = 0
            
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc=f"  Query-Rank{rank}", disable=not is_main_process):
                    batch_dict = batch_to_device(batch, device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(qry=batch_dict)
                    
                    batch_embeddings = output["qry_reps"].cpu().detach().float().numpy()
                    encoded_tensor.append(batch_embeddings)
                    
                    batch_size = batch_embeddings.shape[0]
                    batch_indices = local_indices[idx_pointer:idx_pointer + batch_size]
                    batch_paired_data = [eval_qry_dataset.paired_data[i] for i in batch_indices]
                    encoded_paired_data.extend(batch_paired_data)
                    
                    idx_pointer += batch_size
            
            local_paired_data = encoded_paired_data
            
            if world_size > 1:
                local_embeddings = np.concatenate(encoded_tensor)
                
                all_embeddings = [None] * world_size
                all_indices = [None] * world_size
                all_paired_data = [None] * world_size
                
                dist.all_gather_object(all_embeddings, local_embeddings)
                dist.all_gather_object(all_indices, local_indices)
                dist.all_gather_object(all_paired_data, local_paired_data)
                
                if is_main_process:
                    total_size = len(eval_qry_dataset)
                    final_embeddings = np.zeros((total_size, local_embeddings.shape[1]), dtype=np.float32)
                    final_paired_data = [None] * total_size
                    
                    for embs, indices, paired in zip(all_embeddings, all_indices, all_paired_data):
                        for i, idx in enumerate(indices):
                            final_embeddings[idx] = embs[i]
                            final_paired_data[idx] = paired[i]
                    
                    with open(encode_qry_path, 'wb') as f:
                        pickle.dump((final_embeddings, final_paired_data), f)
                    print(f"    ✓ Saved {len(final_embeddings)} queries")
            else:
                encoded_tensor = np.concatenate(encoded_tensor)
                with open(encode_qry_path, 'wb') as f:
                    pickle.dump((encoded_tensor, local_paired_data), f)
                if is_main_process:
                    print(f"    ✓ Saved {len(encoded_tensor)} queries")
            
            if world_size > 1:
                local_indices = list(tgt_sampler)
            else:
                local_indices = list(range(len(eval_tgt_dataset)))
            
            encoded_tensor = []
            encoded_paired_data = []
            idx_pointer = 0
            
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc=f"  Target-Rank{rank}", disable=not is_main_process):
                    batch_dict = batch_to_device(batch, device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(tgt=batch_dict)
                    
                    batch_embeddings = output["tgt_reps"].cpu().detach().float().numpy()
                    encoded_tensor.append(batch_embeddings)
                    
                    batch_size = batch_embeddings.shape[0]
                    batch_indices = local_indices[idx_pointer:idx_pointer + batch_size]
                    batch_paired_data = [eval_tgt_dataset.paired_data[i] for i in batch_indices]
                    encoded_paired_data.extend(batch_paired_data)
                    
                    idx_pointer += batch_size
            
            local_paired_data = encoded_paired_data
            
            if world_size > 1:
                local_embeddings = np.concatenate(encoded_tensor)
                
                all_embeddings = [None] * world_size
                all_indices = [None] * world_size
                all_paired_data = [None] * world_size
                
                dist.all_gather_object(all_embeddings, local_embeddings)
                dist.all_gather_object(all_indices, local_indices)
                dist.all_gather_object(all_paired_data, local_paired_data)
                
                if is_main_process:
                    total_size = len(eval_tgt_dataset)
                    final_embeddings = np.zeros((total_size, local_embeddings.shape[1]), dtype=np.float32)
                    final_paired_data = [None] * total_size
                    
                    for embs, indices, paired in zip(all_embeddings, all_indices, all_paired_data):
                        for i, idx in enumerate(indices):
                            final_embeddings[idx] = embs[i]
                            final_paired_data[idx] = paired[i]
                    
                    with open(encode_tgt_path, 'wb') as f:
                        pickle.dump((final_embeddings, final_paired_data), f)
                    print(f"    ✓ Saved {len(final_embeddings)} targets")
            else:
                encoded_tensor = np.concatenate(encoded_tensor)
                with open(encode_tgt_path, 'wb') as f:
                    pickle.dump((encoded_tensor, local_paired_data), f)
                if is_main_process:
                    print(f"    ✓ Saved {len(encoded_tensor)} targets")
            
            if world_size > 1:
                dist.barrier()

    if is_main_process:
        all_results = {}
        
        for subset in tqdm(data_args.subset_name, desc="Calculating"):
            encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
            encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
            
            if not os.path.exists(encode_qry_path) or not os.path.exists(encode_tgt_path):
                print(f"⚠️  Skipping {subset}")
                continue
            
            with open(encode_qry_path, 'rb') as f:
                qry_tensor, qry_index = pickle.load(f)
            with open(encode_tgt_path, 'rb') as f:
                tgt_tensor, tgt_index = pickle.load(f)
            
            qry_dict, tgt_dict = {}, {}
            for qry_t, tt in zip(qry_tensor, qry_index):
                text, img_path = tt["text"], tt["img_path"]
                qry_dict[(text, img_path)] = qry_t
            
            for tgt_t, tt in zip(tgt_tensor, tgt_index):
                text, img_path = tt["text"], tt["img_path"]
                tgt_dict[(text, img_path)] = tgt_t

            if is_local:
                eval_data = load_dataset(
                    os.path.join(data_args.dataset_name, subset),
                    split=data_args.dataset_split,
                )
            else:
                eval_data = load_dataset(
                    data_args.dataset_name, subset,
                    split=data_args.dataset_split,
                )
            
            n_correct = 0
            all_pred = []
            
            for row in eval_data:
                qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]
                tgt_t, all_candidates = [], []
                
                for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                    tgt_t.append(tgt_dict[tt])
                    all_candidates.append(tt)
                
                tgt_t = np.stack(tgt_t, axis=0)
                scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
                if pred == 0:
                    n_correct += 1
                all_pred.append(all_candidates[pred])
            
            with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
                for item in all_pred:
                    f.write(f"{item}\n")
            
            accuracy = n_correct / len(eval_data)
            score_dict = {
                "acc": accuracy,
                "num_correct": n_correct,
                "num_pred": len(eval_data)
            }
            
            score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
            with open(score_path, "w") as f:
                json.dump(score_dict, f, indent=4)
            
            all_results[subset] = accuracy
            print(f"  {subset:20s}: {accuracy:.4f} ({n_correct}/{len(eval_data)})")

        print("\n" + "=" * 80)
        print("Final Results")
        print("=" * 80)
        
        for subset, acc in all_results.items():
            print(f"  {subset:20s}: {acc:.4f}")
        
        if all_results:
            avg_acc = sum(all_results.values()) / len(all_results)
            print("-" * 80)
            print(f"  {'Average':20s}: {avg_acc:.4f}")
        
        print("=" * 80)
    
    cleanup_distributed()

if __name__ == "__main__":
    main()