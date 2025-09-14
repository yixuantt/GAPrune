
import os
import dotenv
dotenv.load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

from datetime import datetime
import torch
from tqdm import tqdm
import gc
import json
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np

from loss_utils import HardNegativeNLLLoss

loss_fn = HardNegativeNLLLoss()

def get_args():
    parser = argparse.ArgumentParser(description="Domain Alignment Analysis")
    parser.add_argument('--model_name', type=str, default=None, help="Model name or path")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--max_samples', type=int, default=1000, help="Maximum samples to process")
    parser.add_argument('--use_distributed', action='store_true', default = True, help="Use distributed InfoNCE")
    parser.add_argument('--domain_data_path', type=str, default="data/sample/chem-example-output.json", help="Domain data path")
    parser.add_argument('--general_data_path', type=str, default="data/sample/general-example.json", help="General data path")
    parser.add_argument('--output_dir', type=str, default="output", help="Output directory")
    parser.add_argument('--sparsify_layers', nargs='+', default=['attention','mlp'],
                        help="Only save per-weight arrays for these layer types (e.g., mlp attention linear). Others keep JSON stats only.")
    return parser.parse_args()

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def safe_model_name(model_name):
    """Convert model name to safe filename"""
    import re
    model_name = model_name.split('/')[-1]
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)

class TripletDataset(Dataset):
    """Triplet dataset class with validation"""
    def __init__(self, data, domain_name):
        self.data = data
        self.domain_name = domain_name
        self._validate_data()
    
    def _validate_data(self):
        """Validate triplet data structure"""
        for i, item in enumerate(self.data[:5]):  # Check first 5 items
            assert 'query' in item, f"Missing 'query' in item {i}"
            assert 'positive' in item, f"Missing 'positive' in item {i}"
            assert 'negative' in item, f"Missing 'negative' in item {i}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item['domain'] = self.domain_name
        return item

def get_embeddings(inputs, model):
    """Get embeddings with proper pooling"""
    outputs = model(**inputs, output_hidden_states=False)
    return last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Last token pooling (for decoder-only models like Llama)"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def accumulate_gradients_over_batches(mixed_loader, model, tokenizer, max_length, 
                                     num_batches=None, max_samples=1000, use_distributed=True):
    print(f"\nAccumulating gradient statistics (max {max_samples} samples)...")
    
    # Initialize accumulation variables
    accumulated_g_gen = {}
    accumulated_g_dom = {}  
    conflict_counts = {}
    seen_counts = {}
    cosine_sums = {}
    
    total_loss_gen = 0.0
    total_loss_dom = 0.0
    
    batch_count = 0
    total_samples_processed = 0
    
    with tqdm(total=num_batches, desc="Processing Batch", unit="batch") as pbar:
        for batch_idx, mixed_batch in enumerate(mixed_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
                
            try:
                batch_size = len(mixed_batch['mixed_samples']['query'])
                if total_samples_processed + batch_size > max_samples:
                    break
            except Exception as e:
                print(f"\nError getting batch size: {e}")
                continue
                
            try:
                # Calculate gradients with distributed InfoNCE
                g_gen, loss_gen = get_gradients_from_samples(
                    mixed_batch['gen_samples'], model, tokenizer, max_length, use_distributed)
                model.zero_grad()
                
                g_dom, loss_dom = get_gradients_from_samples(
                    mixed_batch['dom_samples'], model, tokenizer, max_length, use_distributed)  
                model.zero_grad()
                
                # Accumulate gradients and track conflicts per parameter
                for param_name in g_gen.keys():
                    if param_name not in accumulated_g_gen:
                        accumulated_g_gen[param_name] = torch.zeros_like(g_gen[param_name])
                        accumulated_g_dom[param_name] = torch.zeros_like(g_dom[param_name])
                        conflict_counts[param_name] = 0
                        seen_counts[param_name] = 0
                        cosine_sums[param_name] = 0.0
                    
                    accumulated_g_gen[param_name] += g_gen[param_name]
                    accumulated_g_dom[param_name] += g_dom[param_name] 

                    # Per-batch cosine sign to detect conflicts
                    try:
                        gg = g_gen[param_name].reshape(-1).to(dtype=torch.float32)
                        gd = g_dom[param_name].reshape(-1).to(dtype=torch.float32)
                        normg = torch.norm(gg)
                        normd = torch.norm(gd)
                        if normg > 0 and normd > 0:
                            seen_counts[param_name] += 1
                            cos_sim = torch.dot(gg, gd) / (normg * normd + 1e-8)
                            if cos_sim.item() < 0:
                                conflict_counts[param_name] += 1
                            # accumulate per-batch cosine for mean(cos)
                            cosine_sums[param_name] += float(cos_sim.item())
                    except Exception:
                        pass
                
                # Accumulate losses
                total_loss_gen += loss_gen
                total_loss_dom += loss_dom
                
                batch_count += 1
                total_samples_processed += batch_size
                
                pbar.set_postfix({
                    'Samples': f"{total_samples_processed}",
                    'GenLoss': f"{loss_gen:.4f}",
                    'DomLoss': f"{loss_dom:.4f}"
                })
                pbar.update(1)

                del g_gen, g_dom
                if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nError processing Batch {batch_idx + 1}: {str(e)}")
                import traceback
                traceback.print_exc() 
                continue
    
    if batch_count == 0:
        raise ValueError("No batch processed successfully!")
    
    # Calculate average gradients
    print(f"\nComputing average gradients (based on {batch_count} batches)")
    for param_name in accumulated_g_gen.keys():
        accumulated_g_gen[param_name] /= batch_count
        accumulated_g_dom[param_name] /= batch_count
    
    # Calculate average losses
    avg_loss_gen = total_loss_gen / batch_count
    avg_loss_dom = total_loss_dom / batch_count  
    
    print(f"Average Loss - Gen: {avg_loss_gen:.6f}, Dom: {avg_loss_dom:.6f}")
    
    return accumulated_g_gen, accumulated_g_dom, None, avg_loss_gen, avg_loss_dom, None, batch_count, total_samples_processed, conflict_counts, seen_counts, cosine_sums


def get_gradients_from_samples(samples, model, tokenizer, max_length, use_distributed=True):
    """Calculate gradients from samples with optional distributed InfoNCE"""
    model.train()
    model.zero_grad()
    
    # Encode samples
    queries = samples['query'] if isinstance(samples['query'], list) else [samples['query']]
    positives = samples['positive'] if isinstance(samples['positive'], list) else [samples['positive']]
    negatives = samples['negative'] if isinstance(samples['negative'], list) else [samples['negative']]
    
    q_inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    p_inputs = tokenizer(positives, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    n_inputs = tokenizer(negatives, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    
    # Safe device placement: if model parameters are all on a single CUDA device, move inputs there
    param_devices = {p.device for p in model.parameters()}
    if len(param_devices) == 1 and next(iter(param_devices)).type != 'cpu':
        device = next(iter(param_devices))
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        p_inputs = {k: v.to(device) for k, v in p_inputs.items()}
        n_inputs = {k: v.to(device) for k, v in n_inputs.items()}
    
    # Get embeddings
    q_emb = get_embeddings(q_inputs, model)
    p_emb = get_embeddings(p_inputs, model)
    n_emb = get_embeddings(n_inputs, model)
    
    # Optionally use distributed negatives by gathering embeddings across processes
    if use_distributed and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size > 1:
            gather_q = [torch.zeros_like(q_emb) for _ in range(world_size)]
            gather_p = [torch.zeros_like(p_emb) for _ in range(world_size)]
            gather_n = [torch.zeros_like(n_emb) for _ in range(world_size)]
            dist.all_gather(gather_q, q_emb)
            dist.all_gather(gather_p, p_emb)
            dist.all_gather(gather_n, n_emb)
            q_emb = torch.cat(gather_q, dim=0)
            p_emb = torch.cat(gather_p, dim=0)
            n_emb = torch.cat(gather_n, dim=0)

    loss = loss_fn(q_emb, p_emb, n_emb)
    loss.backward()
    
    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad.detach().clone().cpu()
    
    return gradients, loss.item()


def calculate_domain_alignment(g_gen, g_dom, epsilon=1e-8):
    """Calculate domain alignment scores using unit-vector directional similarity.

    For each parameter's gradient, both general and domain gradients are normalized
    to unit vectors first, then the alignment score is computed as their dot
    product (i.e., cosine similarity of unit vectors).
    """
    alignment_scores = {}

    for param_name in g_gen.keys():
        if param_name in g_dom:
            # Flatten and cast to float32 for numerical stability before normalization
            grad_gen = g_gen[param_name].reshape(-1).to(dtype=torch.float32)
            grad_dom = g_dom[param_name].reshape(-1).to(dtype=torch.float32)

            norm_gen = torch.norm(grad_gen)
            norm_dom = torch.norm(grad_dom)

            # Normalize to unit vectors before computing directional similarity
            if norm_gen > epsilon and norm_dom > epsilon:
                grad_gen_unit = grad_gen / (norm_gen + epsilon)
                grad_dom_unit = grad_dom / (norm_dom + epsilon)
                alignment = torch.dot(grad_gen_unit, grad_dom_unit)
                alignment_scores[param_name] = alignment.item()
            else:
                alignment_scores[param_name] = 0.0

    return alignment_scores


def compute_elementwise_alignment(g_gen, g_dom, epsilon=1e-8):
    """Compute per-weight element-wise alignment arrays only.

    alignment_elementwise: element-wise cosine similarity of unit gradients, i.e.,
    cos_sim(unit(g_gen), unit(g_dom)) computed locally for each parameter position.
    
    This represents the local gradient direction alignment at each parameter position.
    Values range from -1 (opposite directions) to +1 (same direction).
    """
    alignment_elementwise = {}

    for param_name in g_gen.keys():
        if param_name not in g_dom:
            continue
        grad_gen = g_gen[param_name].to(dtype=torch.float32)
        grad_dom = g_dom[param_name].to(dtype=torch.float32)
        norm_gen = torch.norm(grad_gen.reshape(-1))
        norm_dom = torch.norm(grad_dom.reshape(-1))
        if norm_gen > epsilon and norm_dom > epsilon:
            grad_gen_unit = grad_gen / (norm_gen + epsilon)
            grad_dom_unit = grad_dom / (norm_dom + epsilon)
            
            # Compute element-wise cosine similarity
            # This represents the local gradient direction alignment at each position
            elementwise_cos_sim = grad_gen_unit * grad_dom_unit
            
            # For better representation of gradient direction differences,
            # we can also compute local neighborhood statistics
            if grad_gen.dim() >= 2:
                # Add local context by smoothing with small neighborhood
                local_alignment = compute_local_gradient_alignment(grad_gen_unit, grad_dom_unit)
                alignment_elementwise[param_name] = local_alignment.cpu()
            else:
                # For 1D tensors, use direct element-wise cosine similarity
                alignment_elementwise[param_name] = elementwise_cos_sim.cpu()
        else:
            alignment_elementwise[param_name] = torch.zeros_like(grad_gen, device='cpu')

    return alignment_elementwise

def compute_local_gradient_alignment(grad_gen_unit, grad_dom_unit, window_size=3):
    """
    Compute local gradient alignment using neighborhood averaging.
    
    This provides a more robust measure of gradient direction differences
    by considering local context around each parameter position.
    """
    if grad_gen_unit.dim() < 2:
        return grad_gen_unit * grad_dom_unit
    
    # Use convolution to compute local average of element-wise cosine similarity
    # This smooths the alignment scores and provides local context
    elementwise_cos_sim = grad_gen_unit * grad_dom_unit
    
    # Apply local averaging using convolution
    # Create a simple averaging kernel
    kernel_size = min(window_size, min(grad_gen_unit.shape))
    if kernel_size > 1:
        # Use 2D average pooling for smoothing
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=grad_gen_unit.device) / (kernel_size * kernel_size)
        
        # Reshape for convolution (add batch and channel dimensions)
        cos_sim_reshaped = elementwise_cos_sim.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution with padding to maintain size
        padding = kernel_size // 2
        smoothed = torch.nn.functional.conv2d(
            cos_sim_reshaped, kernel, padding=padding
        )
        
        return smoothed.squeeze(0).squeeze(0)
    else:
        return elementwise_cos_sim


def save_per_weight_arrays(arrays_dict, method_name, output_root, model_name, args=None):
    """Save per-weight arrays for selected parameters to .npy files and a small JSON index.

    Only saves arrays for parameters that match attention/MLP patterns (configurable via
    --sparsify_layers), and with dimension >= 2.
    """
    os.makedirs(output_root, exist_ok=True)
    safe_model = safe_model_name(model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_root, safe_model)
    os.makedirs(base_dir, exist_ok=True)
    tensor_dir = os.path.join(base_dir, f"{method_name}_{timestamp}_tensors")
    os.makedirs(tensor_dir, exist_ok=True)
    # For downstream loaders that expect a fixed directory name, provide a stable alias
    patterns_to_sparsify = parse_sparsify_layers(getattr(args, 'sparsify_layers', ['attention','mlp'])) if args is not None else set()

    index = {
        'metadata': {
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'tensor_dir': tensor_dir,
            'sparsify_layers': getattr(args, 'sparsify_layers', ['attention','mlp'])
        },
        'tensors': {}
    }

    for param_name, tensor in arrays_dict.items():
        save_this_param = (tensor.dim() >= 2) and should_sparsify_parameter(param_name, patterns_to_sparsify)
        if not save_this_param:
            continue
        try:
            npy_basename = f"{param_name}.npy"
            npy_path = os.path.join(tensor_dir, npy_basename)
            npy_path_alt = os.path.join(tensor_dir, f"{param_name.replace('.', '_')}.npy")
            arr = tensor.detach().to(dtype=torch.float16).numpy()
            with open(npy_path, 'wb') as f:
                np.save(f, arr, allow_pickle=False)
            if npy_path_alt != npy_path and not os.path.exists(npy_path_alt):
                try:
                    os.link(npy_path, npy_path_alt)
                except Exception:
                    with open(npy_path_alt, 'wb') as f2:
                        np.save(f2, arr, allow_pickle=False)
            # Also save aliases without leading 'model.' to match canonicalized lookups
            try:
                if param_name.startswith('model.'):
                    canonical_name = param_name[len('model.') :]
                    npy_path_canon = os.path.join(tensor_dir, f"{canonical_name}.npy")
                    if not os.path.exists(npy_path_canon):
                        try:
                            os.link(npy_path, npy_path_canon)
                        except Exception:
                            with open(npy_path_canon, 'wb') as f3:
                                np.save(f3, arr, allow_pickle=False)
                    npy_path_canon_alt = os.path.join(tensor_dir, f"{canonical_name.replace('.', '_')}.npy")
                    if not os.path.exists(npy_path_canon_alt):
                        try:
                            os.link(npy_path, npy_path_canon_alt)
                        except Exception:
                            with open(npy_path_canon_alt, 'wb') as f4:
                                np.save(f4, arr, allow_pickle=False)
            except Exception:
                pass
            index['tensors'][param_name] = {
                'shape': list(tensor.shape),
                'path': npy_path
            }
        except Exception as e:
            print(f"Warning: failed saving per-weight array for {param_name} ({method_name}): {e}")
        finally:
            if 'arr' in locals():
                del arr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    index_path = os.path.join(base_dir, f"{method_name}_index_{timestamp}.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Saved per-weight tensors for {method_name} to: {tensor_dir}")
    print(f"Index file: {index_path}")
    return tensor_dir, index_path

def print_analysis_summary(results):
    """Print simplified analysis summary focused on alignment scores"""
    print("\n" + "="*60)
    print("DOMAIN ALIGNMENT ANALYSIS SUMMARY")
    print("="*60)
    
    # Processing statistics
    metadata = results['metadata']
    print(f"Processing Statistics:")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Processed batches: {metadata.get('processed_batches', 'N/A')}")
    print(f"  Processed samples: {metadata.get('total_samples_processed', 'N/A')}")
    print(f"  Dataset sizes: General={metadata['general_data_size']:,}, Domain={metadata['domain_data_size']:,}")
    
    # Alignment statistics
    alignments = list(results['alignment_scores'].values())
    total_params = len(alignments)
    
    print(f"\nAlignment Score Statistics:")
    print(f"  Total parameters analyzed: {total_params:,}")
    print(f"  Mean alignment: {sum(alignments)/len(alignments):.4f}")
    print(f"  Max alignment: {max(alignments):.4f}")
    print(f"  Min alignment: {min(alignments):.4f}")
    print(f"  Std deviation: {torch.tensor(alignments).std().item():.4f}")
    
    # Distribution of alignment scores
    positive_aligned = sum(1 for a in alignments if a > 0.1)
    negative_aligned = sum(1 for a in alignments if a < -0.1)
    neutral = total_params - positive_aligned - negative_aligned
    
    print(f"\nAlignment Distribution:")
    print(f"  Positively aligned (>0.1): {positive_aligned:,} ({positive_aligned/total_params*100:.1f}%)")
    print(f"  Neutral (-0.1 to 0.1): {neutral:,} ({neutral/total_params*100:.1f}%)")
    print(f"  Negatively aligned (<-0.1): {negative_aligned:,} ({negative_aligned/total_params*100:.1f}%)")
    
    # Loss statistics
    losses = results['losses']
    print(f"\nLoss Statistics (average):")
    print(f"  General Loss: {losses['general']:.6f}")
    print(f"  Domain Loss: {losses['domain']:.6f}")
    if losses['mixed'] is not None:
        print(f"  Mixed Loss: {losses['mixed']:.6f}")
    else:
        print(f"  Mixed Loss: N/A")
    
    # Top aligned parameters
    sorted_params = sorted(results['alignment_scores'].items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop 10 Most Aligned Parameters (by absolute value):")
    for i, (param_name, score) in enumerate(sorted_params[:10]):
        print(f"  {i+1}. {param_name}: {score:.4f}")

    # Conflict rate statistics (if available)
    if 'conflict_rates' in results and isinstance(results['conflict_rates'], dict) and len(results['conflict_rates']) > 0:
        cr_values = list(results['conflict_rates'].values())
        print(f"\nConflict Rate Statistics:")
        print(f"  Mean conflict rate: {sum(cr_values)/len(cr_values):.4f}")
        print(f"  Max conflict rate: {max(cr_values):.4f}")
        print(f"  Min conflict rate: {min(cr_values):.4f}")
        print(f"  High-conflict params (>0.5): {sum(1 for v in cr_values if v > 0.5)}")

def save_results(results, model_name, output_dir):
    """Save simplified results focused on alignment scores"""
    model_dir = os.path.join(output_dir, safe_model_name(model_name))
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save complete results
    result_file = os.path.join(model_dir, f'alignment_scores_{safe_model_name(model_name)}_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary statistics
    alignments = list(results['alignment_scores'].values())
    summary = {
        'timestamp': timestamp,
        'model_name': model_name,
        'total_parameters': len(alignments),
        'processed_batches': results['metadata'].get('processed_batches', 0),
        'total_samples_processed': results['metadata'].get('total_samples_processed', 0),
        'alignment_statistics': {
            'mean': sum(alignments) / len(alignments),
            'max': max(alignments),
            'min': min(alignments),
            'std': torch.tensor(alignments).std().item(),
            'positive_aligned_count': sum(1 for a in alignments if a > 0.1),
            'negative_aligned_count': sum(1 for a in alignments if a < -0.1)
        },
        'losses': results['losses'],
        'conflict_rates': results.get('conflict_rates', {})
    }

    summary_file = os.path.join(model_dir, f'alignment_summary_{safe_model_name(model_name)}_{timestamp}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"  Alignment scores: {result_file}")
    print(f"  Summary: {summary_file}")

    return result_file, summary_file

class MixedTripletDataLoader:
    """Mixed domain data loader with better error handling"""
    def __init__(self, gen_data, dom_data, batch_size=8):
        self.gen_dataset = TripletDataset(gen_data, 'general')
        self.dom_dataset = TripletDataset(dom_data, 'domain')
        self.batch_size = batch_size
        
        # Ensure balanced batches
        self.gen_batch_size = batch_size // 2
        self.dom_batch_size = batch_size - self.gen_batch_size
        
        # Set drop_last=True to avoid incomplete batches
        self.gen_loader = DataLoader(
            self.gen_dataset, 
            batch_size=self.gen_batch_size, 
            shuffle=True,
            drop_last=True
        )
        self.dom_loader = DataLoader(
            self.dom_dataset, 
            batch_size=self.dom_batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        # Calculate actual number of batches
        self.num_batches = min(len(self.gen_loader), len(self.dom_loader))
        print(f"DataLoader created - Total batches available: {self.num_batches}")
        
    def __len__(self):
        return self.num_batches
        
    def __iter__(self):
        gen_iter = iter(self.gen_loader)
        dom_iter = iter(self.dom_loader)
        
        for _ in range(self.num_batches):
            try:
                gen_batch = next(gen_iter)
                dom_batch = next(dom_iter)
                
                mixed_batch = {
                    'gen_samples': gen_batch,
                    'dom_samples': dom_batch,
                    'mixed_samples': self._combine_batches(gen_batch, dom_batch)
                }
                
                yield mixed_batch
                
            except StopIteration:
                break
            except Exception as e:
                print(f"Error in data loader: {e}")
                continue
    
    def _combine_batches(self, gen_batch, dom_batch):
        """Combine two batches maintaining structure"""
        combined = {}
        for key in ['query', 'positive', 'negative']:
            if key in gen_batch and key in dom_batch:
                gen_items = gen_batch[key]
                dom_items = dom_batch[key]
                
                # Ensure lists
                if not isinstance(gen_items, list):
                    gen_items = gen_items.tolist() if hasattr(gen_items, 'tolist') else list(gen_items)
                if not isinstance(dom_items, list):
                    dom_items = dom_items.tolist() if hasattr(dom_items, 'tolist') else list(dom_items)
                
                combined[key] = gen_items + dom_items
        
        return combined

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
            pass
    return patterns_to_sparsify

def should_sparsify_parameter(param_name, patterns_to_sparsify):
    return any(pat in param_name for pat in patterns_to_sparsify) if patterns_to_sparsify else True

def main():
    """Simplified main function focused on alignment score calculation"""
    print("Starting Domain Alignment Analysis...")
    args = get_args()
    
    print("Analysis Configuration:")
    
    # Model setup
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    for param in model.parameters():
        param.requires_grad = True

    # Load data
    print("\nLoading data...")
    general_data = read_json(args.general_data_path)
    domain_data = read_json(args.domain_data_path)
    
    print(f"General data length: {len(general_data):,}")
    print(f"Domain data length: {len(domain_data):,}")
    
    # Create data loader
    print("\nCreating mixed data loader...")
    mixed_loader = MixedTripletDataLoader(
        general_data, 
        domain_data, 
        batch_size=args.batch_size
    )
    
    # Process batches and accumulate gradients
    print("\nProcessing batches and accumulating gradient statistics...")
    num_batches = args.max_samples // args.batch_size * 2
    
    g_gen, g_dom, g_mix, loss_gen, loss_dom, loss_mix, processed_batches, total_samples, conflict_counts, seen_counts, cosine_sums = accumulate_gradients_over_batches(
        mixed_loader, model, tokenizer, 512, 
        num_batches=num_batches, 
        max_samples=args.max_samples,
        use_distributed=args.use_distributed
    )
    
    # Calculate alignment scores
    print("\nCalculating domain alignment scores...")
    alignment_scores = calculate_domain_alignment(g_gen, g_dom)

    # Compute per-weight element-wise arrays for attention and MLP params
    print("\nComputing per-weight alignment/projection arrays for selected layers...")
    alignment_elem = compute_elementwise_alignment(g_gen, g_dom)

    # Save per-weight arrays to disk similar to a3_pruner_2.py
    try:
        save_per_weight_arrays(alignment_elem, method_name='alignment_elementwise', output_root=args.output_dir, model_name=model_name, args=args)
    except Exception as e:
        print(f"Warning: saving per-weight arrays failed: {e}")

    # Calculate per-parameter conflict rates
    conflict_rates = {}
    for param_name, seen in seen_counts.items():
        if seen > 0:
            conflict_rates[param_name] = conflict_counts.get(param_name, 0) / float(seen)
        else:
            conflict_rates[param_name] = 0.0
    
    # Organize results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name, 
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'use_distributed': args.use_distributed,
            'processed_batches': processed_batches,
            'total_samples_processed': total_samples,
            'general_data_size': len(general_data),
            'domain_data_size': len(domain_data),
            'domain_data_path': args.domain_data_path,
            'general_data_path': args.general_data_path
        },
        'losses': {
            'general': loss_gen,
            'domain': loss_dom, 
            'mixed': loss_mix if loss_mix is not None else None
        },
        'alignment_scores': alignment_scores,
        'conflict_rates': conflict_rates
    }
    
    # Print summary
    print_analysis_summary(results)
    
    # Save results
    print("\nSaving results...")
    result_file, summary_file = save_results(
        results, 
        model_name=model_name.split('/')[-1], 
        output_dir=args.output_dir
    )
    
    print(f"\nAnalysis completed!")
    print(f"Processed {total_samples} samples ({processed_batches} batches)")
    
    return results, result_file, summary_file

if __name__ == "__main__":
    main()