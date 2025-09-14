
import os
import dotenv
dotenv.load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

import torch
from typing import cast, List, Dict, Union, Optional, Set, Tuple
import torch.nn.functional as F
import platform

def get_layer_key(param_name: str) -> str:
    """Heuristic to map parameter name to a layer-level group key."""
    tokens = param_name.split('.')
    if 'layers' in tokens:
        idx = tokens.index('layers')
        if idx + 1 < len(tokens):
            return '.'.join(tokens[:idx + 2])  # e.g., model.layers.0
    if len(tokens) >= 2:
        return '.'.join(tokens[:2])
    return param_name


def parse_sparsify_layers(sparsify_layers: List[str]) -> Set[str]:
    """
    Parse sparsify_layers argument and return set of parameter patterns to sparsify
    """
    layer_patterns = get_layer_type_patterns()
    patterns_to_sparsify = set()

    for layer_type in sparsify_layers:
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
            logger.warning(f"Unknown layer type: {layer_type}")

    return patterns_to_sparsify



def should_sparsify_parameter(param_name: str, patterns_to_sparsify: Set[str]) -> bool:
    """Check if a parameter should be sparsified based on its name and patterns"""
    return any(pattern in param_name for pattern in patterns_to_sparsify)


def canonicalize_param_name(param_name: str) -> str:
    """Strip leading 'model.' prefix used by some HF models to match external stats keys."""
    if param_name.startswith('model.'):
        return param_name[len('model.') :]
    return param_name


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for cross-platform compatibility"""
    invalid_chars = '<>:"|?*' if platform.system() == 'Windows' else ''
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last token embeddings for models with left padding"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for embedding models"""
    return f'Instruct: {task_description}\nQuery: {query}'


def find_alignment_elementwise_dir(base_dir: str) -> Optional[str]:
    """Find directory containing 'alignment_elementwise_' in its name"""
    if not os.path.isdir(base_dir):
        return None
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and 'tensors' in item:
            return item_path
    return None


def get_layer_type_patterns() -> Dict[str, List[str]]:
    """
    Define patterns for different layer types
    """
    return {
        'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'layernorm': ['input_layernorm', 'post_attention_layernorm', 'q_norm', 'k_norm'],
        'embed': ['embed_tokens'],
        'final_norm': ['norm.weight'],

        # Fine-grained MLP components
        'mlp_gate': ['gate_proj'],
        'mlp_up': ['up_proj'],
        'mlp_down': ['down_proj'],

        # Fine-grained attention components
        'attn_q': ['q_proj'],
        'attn_k': ['k_proj'],
        'attn_v': ['v_proj'],
        'attn_o': ['o_proj'],

        # Fine-grained normalization components
        'norm_input': ['input_layernorm'],
        'norm_post_attn': ['post_attention_layernorm'],
        'norm_qk': ['q_norm', 'k_norm'],
    }

def print_available_layer_types():
    """Print available layer types for sparsification"""
    layer_patterns = get_layer_type_patterns()

    print("\n=== Available Layer Types for Sparsification ===")
    print("\n🔸 Main Categories:")
    print("  - mlp: MLP layers (gate_proj, up_proj, down_proj)")
    print("  - attention: Attention layers (q_proj, k_proj, v_proj, o_proj)")
    print("  - layernorm: Normalization layers")
    print("  - embed: Embedding layer")
    print("  - final_norm: Final normalization layer")
    print("  - linear: All linear layers (mlp + attention)")
    print("  - all: All layers (excluding final_norm)")

    print("\n🔸 Fine-grained MLP Components:")
    print("  - mlp_gate: Gate projection only")
    print("  - mlp_up: Up projection only")
    print("  - mlp_down: Down projection only")

    print("\n🔸 Fine-grained Attention Components:")
    print("  - attn_q: Query projection only")
    print("  - attn_k: Key projection only")
    print("  - attn_v: Value projection only")
    print("  - attn_o: Output projection only")