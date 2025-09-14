import os
import dotenv
dotenv.load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

import torch
import torch.distributed as dist
from tqdm import tqdm
import gc
import json
from datetime import datetime
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import argparse
from loss_utils import HardNegativeNLLLoss, cos_sim, mismatched_sizes_all_gather


loss_fn = HardNegativeNLLLoss()

# Difficulty weighting strength for retrieval hardness
DIFFICULTY_ALPHA = 5.0
INBATCH_SCALE = 20.0

def get_args():
    parser = argparse.ArgumentParser(description="Domain Alignment Analysis")
    parser.add_argument('--model_name', type=str, default=None, help="Model name or path")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--max_samples', type=int, default=1000, help="Maximum samples to process")
    parser.add_argument('--output_dir', type=str, default='results', help="Output directory")
    parser.add_argument('--data_path', type=str, default=None, help="Data path")
    parser.add_argument('--sparsify_layers', nargs='+', default=['attention','mlp'],
                        help="Only save per-weight arrays for these layer types (e.g., mlp attention linear). Others keep JSON stats only.")
    return parser.parse_args()

max_length = 512
# --------------------- Layer pattern helpers (match eval script) ---------------------
def get_layer_type_patterns():
    return {
        'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'layernorm': ['input_layernorm', 'post_attention_layernorm', 'q_norm', 'k_norm'],
        'embed': ['embed_tokens'],
        'final_norm': ['norm.weight'],

        'mlp_gate': ['gate_proj'],
        'mlp_up': ['up_proj'],
        'mlp_down': ['down_proj'],

        'attn_q': ['q_proj'],
        'attn_k': ['k_proj'],
        'attn_v': ['v_proj'],
        'attn_o': ['o_proj'],

        'norm_input': ['input_layernorm'],
        'norm_post_attn': ['post_attention_layernorm'],
        'norm_qk': ['q_norm', 'k_norm'],
    }

def parse_sparsify_layers(sparsify_layers):
    layer_patterns = get_layer_type_patterns()
    patterns_to_sparsify = set()
    for layer_type in sparsify_layers or []:
        if layer_type == 'all':
            for key, patterns in layer_patterns.items():
                if key != 'final_norm':
                    patterns_to_sparsify.update(patterns)
        elif layer_type == 'linear':
            patterns_to_sparsify.update(layer_patterns['mlp'])
            patterns_to_sparsify.update(layer_patterns['attention'])
        elif layer_type in layer_patterns:
            patterns_to_sparsify.update(layer_patterns[layer_type])
        else:
            # unknown type ignored
            pass
    return patterns_to_sparsify

def should_sparsify_parameter(param_name, patterns_to_sparsify):
    return any(pat in param_name for pat in patterns_to_sparsify) if patterns_to_sparsify else True


class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query, positive, negative
        query_encoding = self.tokenizer(
            item['query'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            item['positive'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        negative_encoding = self.tokenizer(
            item['negative'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'query': {k: v.squeeze(0) for k, v in query_encoding.items()},
            'positive': {k: v.squeeze(0) for k, v in positive_encoding.items()},
            'negative': {k: v.squeeze(0) for k, v in negative_encoding.items()}
        }

def collate_fn(batch):
    queries = {k: torch.stack([item['query'][k] for item in batch]) for k in batch[0]['query'].keys()}
    positives = {k: torch.stack([item['positive'][k] for item in batch]) for k in batch[0]['positive'].keys()}
    negatives = {k: torch.stack([item['negative'][k] for item in batch]) for k in batch[0]['negative'].keys()}
    
    return queries, positives, negatives

def safe_normalize(embeddings, eps=1e-8):
    norm = embeddings.norm(p=2, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return embeddings / norm


def compute_combined_loss(anchor, positive, negative):
    losses = {}
    
    if torch.isnan(anchor).any() or torch.isnan(positive).any() or torch.isnan(negative).any():
        print("Warning: NaN detected in embeddings!")
        return {
            'infonce': torch.tensor(1e-6, device=anchor.device, requires_grad=True),
            'triplet': torch.tensor(1e-6, device=anchor.device, requires_grad=True),
            'total': torch.tensor(2e-6, device=anchor.device, requires_grad=True)
        }
    
    # 1. InfoNCE Loss
    try:
        infonce_loss = loss_fn(anchor, positive, negative)
        if torch.isnan(infonce_loss):
            print("Warning: NaN in InfoNCE loss")
            infonce_loss = torch.tensor(1e-6, device=anchor.device, requires_grad=True)
    except Exception as e:
        print(f"Error in InfoNCE loss: {e}")
        infonce_loss = torch.tensor(1e-6, device=anchor.device, requires_grad=True)
    
    losses['infonce'] = infonce_loss

    total_loss = infonce_loss
    
    if torch.isnan(total_loss):
        print("Warning: NaN in total loss, returning small value")
        total_loss = torch.tensor(1e-6, device=anchor.device, requires_grad=True)
    
    losses['total'] = total_loss
    
    return losses

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Last token pooling with safety checks"""
    if attention_mask.sum() == 0:
        print("Warning: Empty attention mask detected")
        return last_hidden_states.mean(dim=1)
    
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sequence_lengths = torch.clamp(sequence_lengths, min=0)
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# ---------------------------------------------------------------------
# In-batch InfoNCE utilities with per-sample losses and difficulty weights
# ---------------------------------------------------------------------

def _get_global_offset_and_count(local_count: int, device: torch.device) -> (int, int, int):
    """Compute the global offset for this rank in concatenated all_gather order."""
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_sz = torch.tensor([local_count], device=device, dtype=torch.int64)
        sz_list = [torch.zeros_like(local_sz) for _ in range(world_size)]
        dist.all_gather(sz_list, local_sz)
        sizes = [int(t.item()) for t in sz_list]
        offset = sum(sizes[:rank])
        total = sum(sizes)
        return offset, local_count, total
    return 0, local_count, local_count


def compute_inbatch_per_sample_losses_and_weights(q_emb: Tensor, p_emb: Tensor, scale: float = INBATCH_SCALE, difficulty_alpha: float = DIFFICULTY_ALPHA):
    """Compute per-sample in-batch InfoNCE losses and difficulty weights for local samples.

    Returns:
        local_losses: Tensor [local_batch]
        local_weights: Tensor [local_batch]
    """
    device = q_emb.device
    local_q = q_emb.size(0)

    # all_gather with autograd support if available
    if dist.is_available() and dist.is_initialized():
        full_q_list = mismatched_sizes_all_gather(q_emb)
        full_p_list = mismatched_sizes_all_gather(p_emb)
        q_all = torch.cat(full_q_list, dim=0)
        p_all = torch.cat(full_p_list, dim=0)
    else:
        q_all, p_all = q_emb, p_emb

    # Similarity matrix and labels
    scores = cos_sim(q_all, p_all) * scale  # [Q_all, P_all]
    total_q = scores.size(0)
    labels = torch.arange(total_q, device=scores.device, dtype=torch.long)

    ce = nn.CrossEntropyLoss(reduction='none')
    losses_all = ce(scores, labels)  # [Q_all]

    # Difficulty weights via margin (pos - max_neg)
    pos_logits = scores.diag()
    scores_masked = scores.clone()
    scores_masked.fill_diagonal_(float('-inf'))
    max_neg, _ = scores_masked.max(dim=1)
    margin = pos_logits - max_neg
    weights_all = torch.sigmoid(difficulty_alpha * (-margin))

    # Slice local range
    offset, local_count, _ = _get_global_offset_and_count(local_q, device)
    local_losses = losses_all[offset: offset + local_count]
    local_weights = weights_all[offset: offset + local_count]
    return local_losses, local_weights

# ---------------------------------------------------------------------
# Structure-aware importance computation functions
# ---------------------------------------------------------------------

def compute_structure_fisher_hessian(model, loader, micro_batch_size: int = 8):
    """Per-sample (or micro-batch) Fisher Information estimator with NaN protection.

    Computes Fisher diagonal as average over samples of squared per-sample gradients.
    For micro_batch_size > 1, uses an approximation based on the micro-batch loss.
    """
    model.train()  # enable gradients
    fisher_diag = {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters() if p.requires_grad}
    total_samples = 0

    for batch_idx, (q_in, p_in, n_in) in enumerate(tqdm(loader, desc="Structure Fisher (per-sample)")):
        device = next(model.parameters()).device
        q_in, p_in, n_in = map(lambda d: {k: v.to(device) for k, v in d.items()}, (q_in, p_in, n_in))
        batch_size = q_in["input_ids"].size(0)

        # iterate over samples or micro-batches
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            mb_q = {k: v[start:end] for k, v in q_in.items()}
            mb_p = {k: v[start:end] for k, v in p_in.items()}
            mb_n = {k: v[start:end] for k, v in n_in.items()}

            model.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=(device.type if torch.cuda.is_available() else 'cpu'), dtype=torch.bfloat16):
                    q_out = model(**mb_q)
                    p_out = model(**mb_p)
                    n_out = model(**mb_n)

                    q_emb = last_token_pool(q_out.last_hidden_state, mb_q["attention_mask"])
                    p_emb = last_token_pool(p_out.last_hidden_state, mb_p["attention_mask"])
                    n_emb = last_token_pool(n_out.last_hidden_state, mb_n["attention_mask"])

                    loss_dict = compute_combined_loss(q_emb, p_emb, n_emb)
                    total_loss = loss_dict['total']

                if not torch.isnan(total_loss) and total_loss.item() > 0:
                    total_loss.backward()

                    for n, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            grad = p.grad.detach()
                            if not torch.isnan(grad).any():
                                fisher_diag[n] += (grad.cpu() ** 2)
                            else:
                                print(f"Warning: NaN gradient detected in {n}")
                    total_samples += (end - start)
                else:
                    print("Warning: Invalid loss detected, skipping micro-batch")
            except Exception as e:
                print(f"Error in micro-batch processing: {e}")
                continue
            finally:
                del q_emb, p_emb, n_emb, total_loss, loss_dict

        if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if total_samples > 0:
        for n in fisher_diag:
            fisher_diag[n] /= float(total_samples)
            fisher_diag[n] += 1e-8
    else:
        print("Warning: No valid samples processed")
        for n in fisher_diag:
            fisher_diag[n] += 1e-6

    model.eval()
    return fisher_diag



def save_complete_importance_scores(scores, method_name, output_dir="results", args=None):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{method_name}_complete_scores_{timestamp}.pkl"
    base_no_ext = filename.replace('.pkl', '')
    tensor_dir = base_no_ext + "_tensors"
    os.makedirs(tensor_dir, exist_ok=True)
    
    complete_scores = {}
    summary_stats = {}
    
    # Collect per-layer normalization (z-score and percentile) and difficulty-aware metadata if present
    layer_to_values = {}
    # Determine which params to actually emit per-weight arrays for
    patterns_to_sparsify = parse_sparsify_layers(getattr(args, 'sparsify_layers', ['mlp'])) if args is not None else set(['gate_proj','up_proj','down_proj','q_proj','k_proj','v_proj','o_proj'])

    for param_name, score_tensor in scores.items():
        if torch.isnan(score_tensor).any():
            print(f"Warning: NaN detected in {param_name}, replacing with zeros")
            score_tensor = torch.nan_to_num(score_tensor, nan=0.0)
        # accumulate for per-layer normalization
        layer_name = '.'.join(param_name.split('.')[:2]) if 'layers.' in param_name else param_name.split('.')[0]
        layer_to_values.setdefault(layer_name, []).append(score_tensor.flatten())

        complete_scores[param_name] = {
            'shape': list(score_tensor.shape),
        }
        summary_stats[param_name] = {
            'mean_score': float(score_tensor.mean()),
        }

        # Stream-save per-weight arrays as .npy for memmap loading later (float16) only for prunable layers and 2D+
        save_this_param = (score_tensor.dim() >= 2) and should_sparsify_parameter(param_name, patterns_to_sparsify)
        if not save_this_param:
            continue
        try:
            npy_path = os.path.join(tensor_dir, f"{param_name}.npy")
            npy_path_alt = os.path.join(tensor_dir, f"{param_name.replace('.', '_')}.npy")
            arr = score_tensor.detach().cpu().to(dtype=torch.float16).numpy()
            with open(npy_path, 'wb') as f:
                np.save(f, arr, allow_pickle=False)
            if npy_path_alt != npy_path and not os.path.exists(npy_path_alt):
                try:
                    os.link(npy_path, npy_path_alt)
                except Exception:
                    with open(npy_path_alt, 'wb') as f2:
                        np.save(f2, arr, allow_pickle=False)
        except Exception as e:
            print(f"Warning: failed saving per-weight array for {param_name}: {e}")
        finally:
            if 'arr' in locals():
                del arr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    # Build per-layer stats
    per_layer_stats = {}
    for layer, tensors in layer_to_values.items():
        cat_vals = torch.cat([t for t in tensors])
        mean = float(cat_vals.mean())
        std = float(cat_vals.std())
        per_layer_stats[layer] = {
            'mean': mean,
            'std': std if std > 1e-12 else 1e-12,
            'count': int(cat_vals.numel())
        }
    
    output_data = {
        'metadata': {
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'total_parameters': sum(scores[n].numel() for n in scores),
            'model_name': args.model_name
        },
        'summary_statistics': summary_stats,
        'complete_scores': complete_scores,
        'per_layer_statistics': per_layer_stats
    }
    
    
    with open(base_no_ext + '.json', 'w') as f:
        json.dump(output_data, f)

    compact_filename = f"{output_dir}/{method_name}_summary_{timestamp}.pkl"
    compact_data = {
        'metadata': output_data['metadata'],
        'summary_statistics': summary_stats
    }
    with open(compact_filename, 'wb') as f:
        pickle.dump(compact_data, f)
    print(f"Summary statistics saved to: {compact_filename}")
    
    return filename, compact_filename

def analyze_importance_scores(scores, method_name):
    print(f"\n{'='*60}")
    print(f"{method_name} Structure-Aware Importance Analysis")
    print(f"{'='*60}")
    
    total_params = 0
    param_stats = []
    
    for param_name, score_tensor in scores.items():
        if torch.isnan(score_tensor).any():
            print(f"Warning: NaN in {param_name}, using nan_to_num")
            score_tensor = torch.nan_to_num(score_tensor, nan=0.0)
        
        mean_score = float(score_tensor.mean())
        std_score = float(score_tensor.std())
        param_count = score_tensor.numel()
        total_params += param_count
        
        param_stats.append({
            'name': param_name,
            'mean_score': mean_score,
            'std_score': std_score,
            'param_count': param_count,
            'total_score': float(score_tensor.sum())
        })
    
    # Sort by mean importance score (descending)
    param_stats.sort(key=lambda x: x['mean_score'], reverse=True)
    
    print(f"Total parameters analyzed: {total_params:,}")
    print(f"Number of parameter groups: {len(param_stats)}")
    
    print(f"\nTop 10 Most Important Parameter Groups:")
    print(f"{'Rank':<4} {'Parameter Name':<45} {'Mean Score':<12} {'Param Count':<12}")
    print("-" * 75)
    
    for i, stat in enumerate(param_stats[:10]):
        print(f"{i+1:<4} {stat['name']:<45} {stat['mean_score']:<12.6f} {stat['param_count']:<12,}")
    
    print(f"\nBottom 5 Least Important Parameter Groups:")
    print(f"{'Rank':<4} {'Parameter Name':<45} {'Mean Score':<12} {'Param Count':<12}")
    print("-" * 75)
    
    for i, stat in enumerate(param_stats[-5:]):
        rank = len(param_stats) - 4 + i
        print(f"{rank:<4} {stat['name']:<45} {stat['mean_score']:<12.6f} {stat['param_count']:<12,}")
    
    return param_stats

def main():
    print("Starting Parameter Importance Analysis")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("Starting Domain Alignment Analysis...")
    args = get_args()
    BATCH_SIZE = args.batch_size

    model_name = args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    for param in model.parameters():
        param.requires_grad = True

    data_path = args.data_path
    # Load data
    print(f"\n Loading data from: {data_path}")
    
    with open(data_path, "r") as f:
        data = json.load(f)

    if args.max_samples < len(data):
        data = data[:args.max_samples]  # Use subset for testing

    print(f"Using {len(data)} samples for analysis")
    
    # Create dataset and dataloader
    dataset = TripletDataset(data, tokenizer, max_length)
    importance_batch_size = min(BATCH_SIZE, len(data))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=importance_batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Created DataLoader with batch_size={importance_batch_size}")
    print(f"Total batches: {len(dataloader)}")

    sample_batch = next(iter(dataloader))
    q_in, p_in, n_in = sample_batch
    q_in, p_in, n_in = map(lambda d: {k: v.to(model.device) for k, v in d.items()}, (q_in, p_in, n_in))
    
    with torch.no_grad():
        q_emb = last_token_pool(model(**q_in).last_hidden_state, q_in["attention_mask"])
        p_emb = last_token_pool(model(**p_in).last_hidden_state, p_in["attention_mask"])
        n_emb = last_token_pool(model(**n_in).last_hidden_state, n_in["attention_mask"])
        
        loss_dict = compute_combined_loss(q_emb, p_emb, n_emb)
        
        print("Loss Component Analysis:")
        print(f"  InfoNCE Loss: {loss_dict['infonce'].item():.4f}")

        if loss_dict['total'].item() > 0:
            infonce_ratio = (loss_dict['infonce'] / loss_dict['total'] * 100).item()
            print(f"  InfoNCE contribution: {infonce_ratio:.1f}%")
    
    results = {}
    
    try:
        domain = data_path.split("/")[-1].split("_")[0].split(".")[0]

        # 2. Structure-aware Fisher Information
        print(f"\nComputing Structure-Aware Fisher Information...")
        fisher_scores = compute_structure_fisher_hessian(model, dataloader)
        results['structure_fisher'] = analyze_importance_scores(fisher_scores, "Structure-Aware Fisher")
        fisher_files = save_complete_importance_scores(fisher_scores, f"{domain}_fisher", args.output_dir, args)
        
        # Summary comparison
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY: Parameter Importance")
        print(f"{'='*70}")
        
        print(f"{'Method':<25} {'Top Parameter Group':<30} {'Mean Score':<15}")
        print("-" * 70)
        
        for method, stats in results.items():
            if stats:
                top_param = stats[0]
                short_name = top_param['name'].split('.')[-2:] if '.' in top_param['name'] else [top_param['name']]
                short_name = '.'.join(short_name)
                print(f"{method:<25} {short_name:<30} {top_param['mean_score']:<15.6f}")
        
        print(f"\nComplete importance scores saved with all parameter values")
        print(f"Total parameters analyzed: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    except Exception as e:
        print(f"Error during computation: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nAnalysis complete! Structure-preserving importance scores computed.")

if __name__ == "__main__":
    main()