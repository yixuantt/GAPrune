import os
import dotenv
dotenv.load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

import json
import torch
import numpy as np
from tqdm import tqdm
from typing import cast, List, Dict, Optional, Set, Tuple, Union
import logging
import argparse
from transformers import AutoTokenizer, AutoModel, is_torch_npu_available
import torch.nn.functional as F

import gc

from utils import (
    find_alignment_elementwise_dir, 
    get_layer_key, parse_sparsify_layers, 
    should_sparsify_parameter, 
    canonicalize_param_name,
    print_available_layer_types,
    sanitize_filename,
    last_token_pool
)

def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()



class PerWeightArrayStore:
    """Lazy loader for per-weight score arrays stored as .npy files alongside a JSON manifest."""

    def __init__(self, json_path: Optional[str]) -> None:
        self.json_path = json_path
        self.available: bool = False
        self.tensor_dir: Optional[str] = None
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self._init_from_json()

    def _init_from_json(self) -> None:
        if not self.json_path:
            return
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)

            logger.info(f"Loading JSON from {self.json_path}")
            logger.debug(f"JSON structure keys: {list(data.keys())}")

            if 'alignment_scores' in data:
                base, _ = os.path.splitext(self.json_path)
                base_dir = os.path.dirname(self.json_path)
                
                # First try to find directory containing 'alignment_elementwise_'
                guess_dir = find_alignment_elementwise_dir(base_dir)
                
                if guess_dir and os.path.isdir(guess_dir):
                    self.tensor_dir = guess_dir
                    self.available = True
                    logger.debug(f"Found tensor directory: {guess_dir}")
                else:
                    guess_dir = base + '_tensors'
                    if os.path.isdir(guess_dir):
                        self.tensor_dir = guess_dir
                        self.available = True
                        logger.debug(f"Found tensor directory: {guess_dir}")
            elif "mean_batch_cosines" in data:
                base, _ = os.path.splitext(self.json_path)
                base_dir = os.path.dirname(self.json_path)
                
                # First try to find directory containing 'alignment_elementwise_'
                guess_dir = find_alignment_elementwise_dir(base_dir)
                
                if guess_dir and os.path.isdir(guess_dir):
                    self.tensor_dir = guess_dir
                    self.available = True
                    logger.debug(f"Found tensor directory: {guess_dir}")
                else:
                    guess_dir = base + '_tensors'
                    if os.path.isdir(guess_dir):
                        self.tensor_dir = guess_dir
                        self.available = True
                        logger.debug(f"Found tensor directory: {guess_dir}")
            elif 'projection_scores' in data:
                base, _ = os.path.splitext(self.json_path)
                base_dir = os.path.dirname(self.json_path)
                
                # First try to find directory containing 'alignment_elementwise_'
                guess_dir = find_alignment_elementwise_dir(base_dir)
                
                if guess_dir and os.path.isdir(guess_dir):
                    self.tensor_dir = guess_dir
                    self.available = True
                    logger.debug(f"Found tensor directory: {guess_dir}")
                else:
                    guess_dir = base + '_tensors'
                    if os.path.isdir(guess_dir):
                        self.tensor_dir = guess_dir
                        self.available = True
                        logger.debug(f"Found tensor directory: {guess_dir}")
            elif 'conflict_rates' in data:
                base, _ = os.path.splitext(self.json_path)
                base_dir = os.path.dirname(self.json_path)
                
                # First try to find directory containing 'alignment_elementwise_'
                guess_dir = find_alignment_elementwise_dir(base_dir)
                
                if guess_dir and os.path.isdir(guess_dir):
                    self.tensor_dir = guess_dir
                    self.available = True
                    logger.debug(f"Found tensor directory: {guess_dir}")
                else:
                    guess_dir = base + '_tensors'
                    if os.path.isdir(guess_dir):
                        self.tensor_dir = guess_dir
                        self.available = True
                        logger.debug(f"Found tensor directory: {guess_dir}")
            elif 'summary_statistics' in data:
                base, _ = os.path.splitext(self.json_path)
                guess_dir = base + '_tensors'
                if os.path.isdir(guess_dir):
                    self.tensor_dir = guess_dir
                    self.available = True
                    logger.debug(f"Found tensor directory: {guess_dir}")
            else:
                logger.warning(f"Unknown JSON format: {self.json_path}")
                logger.debug(f"Available keys: {list(data.keys())}")
        except Exception as e:
            logger.warning(f"Failed to initialize PerWeightArrayStore: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            self.available = False
            self.tensor_dir = None

    def _candidate_paths(self, param_key: str) -> List[str]:
        key = canonicalize_param_name(param_key)
        if not self.tensor_dir:
            return []
        cand = []
        sanitized_key = sanitize_filename(key)
        cand.append(os.path.join(self.tensor_dir, f"{sanitized_key}.npy"))
        cand.append(os.path.join(self.tensor_dir, f"{sanitized_key.replace('.', '_')}.npy"))
        cand.append(os.path.join(self.tensor_dir, f"{sanitized_key.replace('.', '__')}.npy"))
        return cand

    def get_array_memmap(self, param_key: str) -> Optional[np.memmap]:
        if not self.available:
            return None
        for p in self._candidate_paths(param_key):
            if os.path.isfile(p):
                try:
                    arr = np.load(p, mmap_mode='r')
                    return arr
                except Exception as e:
                    logger.debug(f"Failed to load array from {p}: {e}")
                    continue
        return None


def compute_dai_importance(
    domain_fisher: torch.Tensor,
    general_fisher: torch.Tensor,
    alignment: torch.Tensor,
    param_magnitude: torch.Tensor,
    beta: float = 1.0,
    compression_weight: float = 0.3
) -> torch.Tensor:
    eps = 1e-8
    task_relevance = domain_fisher * param_magnitude
    input_complexity = general_fisher * param_magnitude
    alignment_factor = torch.clamp(1.0 + 0.2 * alignment, 0.8, 1.2)

    ib_objective = task_relevance - beta * input_complexity
    modulated_objective = ib_objective * alignment_factor
    regularization = compression_weight * torch.sqrt(param_magnitude + eps)
    raw_importance = modulated_objective + regularization
    importance = torch.nn.functional.softplus(raw_importance)
    return importance


def compute_importance_scores(
    method: str,
    general_fisher_scores: Dict[str, float],
    domain_fisher_scores: Dict[str, float],
    alignment_scores: Dict[str, float],
    general_fisher_store: PerWeightArrayStore,
    domain_fisher_store: PerWeightArrayStore,
    alignment_store: PerWeightArrayStore,
    param_name: str,
    param: torch.nn.Parameter,
    data_source_stats: Dict = None,
    **kwargs
) -> torch.Tensor:
    """
    Unified interface for computing importance scores using Information Bottleneck method
    """
    device = param.device
    numel = param.numel()

    canonical_name = canonicalize_param_name(param_name)

    domain_fisher = None
    general_fisher = None
    alignment = None

    data_source_used = {
        'domain_fisher': 'fallback',
        'general_fisher': 'fallback',
        'alignment': 'fallback'
    }

    if domain_fisher_store and domain_fisher_store.available:
        domain_fisher_array = domain_fisher_store.get_array_memmap(param_name)
        if domain_fisher_array is not None:
            try:
                array_tensor = torch.tensor(np.asarray(domain_fisher_array), dtype=torch.float32)
                if array_tensor.numel() == numel:
                    domain_fisher = array_tensor.reshape(-1).to(device)
                    data_source_used['domain_fisher'] = 'per_weight_array'
                    logger.debug(f"✓ Loaded per-weight domain Fisher for {param_name}: shape={array_tensor.shape}, mean={array_tensor.mean():.6f}, std={array_tensor.std():.6f}")
                else:
                    logger.warning(f"✗ Domain Fisher array size mismatch for {param_name}: expected {numel}, got {array_tensor.numel()}")
            except Exception as e:
                logger.debug(f"✗ Failed to load domain Fisher array for {param_name}: {e}")

    if domain_fisher is None:
        domain_fisher_scalar = float(domain_fisher_scores.get(canonical_name, 1.0))
        domain_fisher = torch.full((numel,), domain_fisher_scalar, device=device, dtype=torch.float32)
        logger.info(f"⚠ Using fallback domain Fisher for {param_name}: scalar={domain_fisher_scalar:.6f}")

    if general_fisher_store and general_fisher_store.available:
        general_fisher_array = general_fisher_store.get_array_memmap(param_name)
        if general_fisher_array is not None:
            try:
                array_tensor = torch.tensor(np.asarray(general_fisher_array), dtype=torch.float32)
                if array_tensor.numel() == numel:
                    general_fisher = array_tensor.reshape(-1).to(device)
                    data_source_used['general_fisher'] = 'per_weight_array'
                    logger.debug(f"✓ Loaded per-weight general Fisher for {param_name}: shape={array_tensor.shape}, mean={array_tensor.mean():.6f}, std={array_tensor.std():.6f}")
                else:
                    logger.warning(f"✗ General Fisher array size mismatch for {param_name}: expected {numel}, got {array_tensor.numel()}")
            except Exception as e:
                logger.debug(f"✗ Failed to load general Fisher array for {param_name}: {e}")

    if general_fisher is None:
        general_fisher_scalar = float(general_fisher_scores.get(canonical_name, 1.0))
        general_fisher = torch.full((numel,), general_fisher_scalar, device=device, dtype=torch.float32)
        logger.debug(f"⚠ Using fallback general Fisher for {param_name}: scalar={general_fisher_scalar:.6f}")

    if alignment_store and alignment_store.available:
        alignment_array = alignment_store.get_array_memmap(param_name)
        if alignment_array is not None:
            try:
                array_tensor = torch.from_numpy(np.asarray(alignment_array)).float()
                if array_tensor.numel() == numel:
                    alignment = array_tensor.reshape(-1).to(device)
                    data_source_used['alignment'] = 'per_weight_array'
                    logger.debug(f"✓ Loaded per-weight alignment for {param_name}: shape={array_tensor.shape}, mean={array_tensor.mean():.6f}, std={array_tensor.std():.6f}, range=[{array_tensor.min():.6f}, {array_tensor.max():.6f}]")
                else:
                    logger.warning(f"✗ Alignment array size mismatch for {param_name}: expected {numel}, got {array_tensor.numel()}")
            except Exception as e:
                logger.debug(f"✗ Failed to load alignment array for {param_name}: {e}")

    eps = 1e-8

    if alignment is None:
        alignment_scalar = float(alignment_scores.get(canonical_name, 0.0))
        weight_flat = param.data.reshape(-1)
        if len(weight_flat) > 1:
            weight_grad = torch.zeros_like(weight_flat)
            weight_grad[:-1] = torch.abs(weight_flat[1:] - weight_flat[:-1])
            weight_grad[-1] = weight_grad[-2] if len(weight_grad) > 1 else 0
            grad_norm = weight_grad / (weight_grad.mean() + eps)
            alignment = alignment_scalar + 0.2 * (grad_norm - 1.0) * alignment_scalar
        else:
            alignment = torch.full((numel,), alignment_scalar, device=device, dtype=torch.float32)
        alignment = torch.clamp(alignment, -1.0, 1.0)
        logger.debug(f"⚠ Using fallback alignment for {param_name}: scalar={alignment_scalar:.6f} with variation")

    if data_source_stats is not None:
        for metric, source in data_source_used.items():
            data_source_stats[metric][source] += 1

    param_magnitude = param.data.abs().reshape(-1).float()

    if method != 'dai':
        raise ValueError(f"Only 'dai' is supported. Got: {method}")

    importance = compute_dai_importance(
        domain_fisher, general_fisher, alignment, param_magnitude,
        beta=kwargs.get('beta', 1.0),
        compression_weight=kwargs.get('compression_weight', 0.5)
    )

    score_std = importance.std()
    if score_std < 1e-6:
        logger.warning(f"Low variance in importance scores for {param_name}, adding structured variation")
        weight_based_variation = param_magnitude / (param_magnitude.mean() + eps)
        importance = importance * (0.5 + 0.5 * weight_based_variation)

    score_postproc = kwargs.get('score_postproc', 'zscore_sigmoid')
    score_temperature = float(kwargs.get('score_temperature', 1.0) or 1.0)
    try:
        if score_postproc == 'zscore_sigmoid':
            mean = importance.mean()
            std = importance.std() + eps
            z = (importance - mean) / std
            t = max(1e-6, score_temperature)
            importance_pp = torch.sigmoid(z / t)
        elif score_postproc == 'minmax':
            vmin = importance.min()
            vmax = importance.max()
            importance_pp = (importance - vmin) / (vmax - vmin + eps)
        elif score_postproc == 'rank':
            n = importance.numel()
            order = torch.argsort(importance, dim=0)
            ranks = torch.empty_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(1, n + 1, device=importance.device, dtype=torch.float32)
            importance_pp = ranks / float(n)
        else:
            importance_pp = importance
    except Exception as e:
        logger.warning(f"Score post-processing failed for {param_name}: {e}")
        importance_pp = importance

    fusion_alpha = float(kwargs.get('fusion_alpha', 0.0) or 0.0)
    fusion_beta = float(kwargs.get('fusion_beta', 1.0) or 1.0)
    try:
        if fusion_alpha != 0.0 or fusion_beta != 1.0:
            fish_mag = domain_fisher * param_magnitude
            importance = fusion_alpha * fish_mag + fusion_beta * importance_pp
        else:
            importance = importance_pp
    except Exception as e:
        logger.warning(f"Score fusion failed for {param_name}: {e}")
        importance = importance_pp

    return importance


class QwenSparseModel:
    def __init__(
        self,
        model_name_or_path: str = None,
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        general_fisher_scores: str = None,
        domain_fisher_scores: str = None,
        alignment_scores: str = None,
        sparsify_layers: List[str] = None,
        # Unstructured pruning
        unstructured_pct: float = 0.0,
        unstructured_scoring: str = 'dai',
        # allocation & postproc options
        layer_budget_strategy: str = 'global',
        score_postproc: str = 'zscore_sigmoid',
        score_temperature: float = 1.0,
        fusion_alpha: float = 0.0,
        fusion_beta: float = 1.0,
        rho_min: float = 0.05,
        rho_max: float = 0.95,
        # Information Bottleneck specific hyperparameters
        beta: float = 1.0,
        compression_weight: float = 0.5,
    ) -> None:
        """Initialize Qwen Sparse Model for FinMTEB evaluation with Information Bottleneck pruning"""
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.general_fisher_scores = general_fisher_scores
        self.domain_fisher_scores = domain_fisher_scores
        self.alignment_scores = alignment_scores
        self.sparsify_layers = sparsify_layers or ['mlp']
        self.unstructured_pct = unstructured_pct or 0.0
        self.unstructured_scoring = unstructured_scoring
        self.layer_budget_strategy = layer_budget_strategy
        self.score_postproc = score_postproc
        self.score_temperature = score_temperature
        self.fusion_alpha = fusion_alpha
        self.fusion_beta = fusion_beta
        self.layer_rho_min = rho_min
        self.layer_rho_max = rho_max

        self.algorithm_hyperparams = {
            'beta': beta,
            'compression_weight': compression_weight,
        }

        self.data_source_stats = {
            'domain_fisher': {'per_weight_array': 0, 'fallback': 0},
            'general_fisher': {'per_weight_array': 0, 'fallback': 0},
            'alignment': {'per_weight_array': 0, 'fallback': 0}
        }

        self.load_model()

        if self.unstructured_pct and self.unstructured_pct > 0.0:
            self.apply_pruning()

        self.setup_device()

    def load_model(self):
        """Load tokenizer and model"""
        logger.info(f"Loading model from: {self.model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side='left',
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        logger.info("Model and tokenizer loaded successfully")

    def _iter_target_parameters(self, patterns_to_sparsify: Set[str]):
        """Iterate over parameters that should be sparsified"""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 0:
                continue
            if param.dim() < 2:
                continue
            if should_sparsify_parameter(name, patterns_to_sparsify):
                yield name, param

    @torch.no_grad()
    def apply_pruning(self):
        """Apply pruning using the Information Bottleneck method"""
        patterns_to_sparsify = parse_sparsify_layers(self.sparsify_layers)
        logger.info(
            f"Applying {self.unstructured_scoring.upper()} pruning: pct={self.unstructured_pct:.2%}, "
            f"layers={self.sparsify_layers}"
        )

        general_fisher_means: Dict[str, float] = {}
        domain_fisher_means: Dict[str, float] = {}
        alignment_means: Dict[str, float] = {}

        general_fisher_store = PerWeightArrayStore(self.general_fisher_scores)
        domain_fisher_store = PerWeightArrayStore(self.domain_fisher_scores)
        alignment_store = PerWeightArrayStore(self.alignment_scores)

        if self.general_fisher_scores and os.path.exists(self.general_fisher_scores):
            try:
                logger.info(f"Loading JSON from {self.general_fisher_scores}")
                with open(self.general_fisher_scores, 'r') as f:
                    general_fisher_json = json.load(f)
                general_fisher_means = {
                    canonicalize_param_name(k): float(v.get('mean_score', 1.0))
                    for k, v in (general_fisher_json.get('summary_statistics', {}) or {}).items()
                }
            except Exception as e:
                logger.warning(f"Failed to load general_fisher_scores means from {self.general_fisher_scores}: {e}")
        if self.domain_fisher_scores and os.path.exists(self.domain_fisher_scores):
            try:
                logger.info(f"Loading JSON from {self.domain_fisher_scores}")
                with open(self.domain_fisher_scores, 'r') as f:
                    domain_fisher_json = json.load(f)
                domain_fisher_means = {
                    canonicalize_param_name(k): float(v.get('mean_score', 1.0))
                    for k, v in (domain_fisher_json.get('summary_statistics', {}) or {}).items()
                }
            except Exception as e:
                logger.warning(f"Failed to load domain_fisher_scores means from {self.domain_fisher_scores}: {e}")
        if self.alignment_scores and os.path.exists(self.alignment_scores):
            try:
                logger.info(f"Loading JSON from {self.alignment_scores}")
                with open(self.alignment_scores, 'r') as f:
                    alignment_json = json.load(f)

                selected_key = 'alignment_scores'
                selected_dict = None

                if isinstance(alignment_json.get(selected_key, None), dict):
                    selected_dict = alignment_json[selected_key]
                    logger.info(f"Using alignment metric '{selected_key}' from alignment file")
                elif 'summary_statistics' in alignment_json:
                    selected_dict = alignment_json['summary_statistics']
                    logger.info("Using 'summary_statistics' from alignment file")
                elif 'alignment_scores' in alignment_json:
                    selected_dict = alignment_json['alignment_scores']
                    logger.info("Using 'alignment_scores' from alignment file")

                if isinstance(selected_dict, dict):
                    if selected_key == 'conflict_rates':
                        alignment_means = {
                            canonicalize_param_name(k): float(1.0 - 2.0 * float(v))
                            for k, v in selected_dict.items()
                        }
                    else:
                        alignment_means = {
                            canonicalize_param_name(k): float(v.get('mean_score', 0.0) if isinstance(v, dict) else v)
                            for k, v in selected_dict.items()
                        }
            except Exception as e:
                logger.warning(f"Failed to load alignment_scores from {self.alignment_scores}: {e}")

        total_candidates = 0
        candidates: List[Tuple[str, torch.nn.Parameter]] = []
        layer_groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}
        for name, param in self._iter_target_parameters(patterns_to_sparsify):
            numel = int(param.data.numel())
            if numel <= 0:
                continue
            candidates.append((name, param))
            total_candidates += numel
            lk = get_layer_key(name)
            layer_groups.setdefault(lk, []).append((name, param))

        if total_candidates == 0:
            logger.info("No candidate parameters found for pruning. Skipping.")
            return

        target_k = int(total_candidates * float(self.unstructured_pct))
        if target_k <= 0:
            logger.info("Requested unstructured_pct too small; nothing to prune.")
            return

        per_param_scores: Dict[str, torch.Tensor] = {}
        layer_sizes: Dict[str, int] = {}

        for lk, plist in tqdm(layer_groups.items(), desc=f"Scoring per layer ({self.unstructured_scoring})"):
            layer_total = 0
            for name, param in plist:
                weight = param.data
                numel = int(weight.numel())
                if numel <= 0:
                    continue
                importance_scores = compute_importance_scores(
                    self.unstructured_scoring,
                    general_fisher_means,
                    domain_fisher_means,
                    alignment_means,
                    general_fisher_store,
                    domain_fisher_store,
                    alignment_store,
                    name,
                    param,
                    data_source_stats=self.data_source_stats,
                    score_postproc=self.score_postproc,
                    score_temperature=self.score_temperature,
                    fusion_alpha=self.fusion_alpha,
                    fusion_beta=self.fusion_beta,
                    **self.algorithm_hyperparams
                )
                per_param_scores[name] = importance_scores.cpu()
                layer_total += numel
            layer_sizes[lk] = layer_total

        target_k = int(total_candidates * float(self.unstructured_pct))
        if target_k <= 0:
            logger.info("Requested unstructured_pct too small; nothing to prune after scoring.")
            return

        strategy = getattr(self, 'layer_budget_strategy', 'global')
        masked_total = 0
        stats = {
            'method': self.unstructured_scoring,
            'unstructured_pct_requested': float(self.unstructured_pct),
            'estimated_threshold': None,
            'scoring': self.unstructured_scoring,
            'sparsify_layers': self.sparsify_layers,
            'total_params_considered': int(total_candidates),
            'masked_params': 0,
            'hyperparameters': {},
            'per_param': {},
            'per_layer_allocations': {},
        }

        if strategy == 'global':
            concat_scores = []
            offsets = {}
            running = 0
            for name, _ in candidates:
                if name not in per_param_scores:
                    continue
                s = per_param_scores[name]
                offsets[name] = (running, running + s.numel())
                running += s.numel()
                concat_scores.append(s)

            if not concat_scores:
                logger.info("No scores collected; skipping pruning.")
                return

            global_scores = torch.cat(concat_scores, dim=0)

            kth = max(1, min(target_k, global_scores.numel()-1))
            vals, idxs = torch.topk(global_scores, kth, largest=False)
            global_threshold = float(vals.max().item())
            topk_indices = idxs

            per_layer_masked = {lk: 0 for lk in layer_groups.keys()}

            for lk, plist in layer_groups.items():
                for name, param in plist:
                    if name not in per_param_scores:
                        continue
                    weight = param.data
                    importance_scores = per_param_scores[name].to(weight.device)
                    numel = int(weight.numel())
                    score_min = float(importance_scores.min().item())
                    score_max = float(importance_scores.max().item())
                    score_mean = float(importance_scores.mean().item())
                    score_std = float(importance_scores.std().item())

                    start_offset, end_offset = offsets[name]
                    in_this = (topk_indices >= start_offset) & (topk_indices < end_offset)
                    local_indices = (topk_indices[in_this] - start_offset).to(dtype=torch.long)
                    local_indices = local_indices.to(weight.device)
                    local_mask = torch.zeros(numel, dtype=torch.bool, device=weight.device)
                    if local_indices.numel() > 0:
                        local_mask[local_indices] = True
                        flat_view = weight.view(-1)
                        flat_view[local_mask] = 0
                    masked = int(local_mask.sum().item())
                    mask_ratio = float(masked / numel) if numel > 0 else 0.0
                    masked_total += masked
                    per_layer_masked[lk] += masked

                    stats['per_param'][name] = {
                        'numel': int(numel),
                        'masked': masked,
                        'ratio': (0.0 if numel == 0 else mask_ratio),
                        'mean_importance': score_mean,
                        'score_stats': {
                            'min': score_min,
                            'max': score_max,
                            'mean': score_mean,
                            'std': score_std,
                            'threshold': float(global_threshold),
                            'masked_ratio': mask_ratio
                        }
                    }

            for lk in layer_groups.keys():
                stats['per_layer_allocations'][lk] = {
                    'masked': int(per_layer_masked[lk]),
                    'layer_size': int(layer_sizes[lk]),
                    'threshold': global_threshold,
                }
        else:
            sizes = torch.tensor([max(1, layer_sizes[lk]) for lk in layer_groups.keys()], dtype=torch.float64)
            weights = sizes.double() / sizes.double().sum()
            per_layer_k = (weights * target_k).long()
            deficit = target_k - int(per_layer_k.sum().item())
            if deficit != 0:
                order = torch.argsort(-weights)
                for idx in order[:abs(deficit)].tolist():
                    per_layer_k[idx] += 1 if deficit > 0 else -1

            rho_min = getattr(self, 'layer_rho_min', 0.05)
            rho_max = getattr(self, 'layer_rho_max', 0.95)
            for i in range(len(per_layer_k)):
                n = int(sizes[i].item())
                kmin = int(max(0, np.floor(rho_min * n)))
                kmax = int(max(0, np.floor(rho_max * n)))
                per_layer_k[i] = torch.clamp(per_layer_k[i], kmin, kmax)

            lks = list(layer_groups.keys())
            for lk_idx, lk in enumerate(lks):
                target_layer_k = int(per_layer_k[lk_idx].item())
                plist = layer_groups[lk]
                scores_list = [per_param_scores[name] for name, _ in plist if name in per_param_scores]
                if not scores_list:
                    continue
                layer_scores = torch.cat(scores_list, dim=0)
                if target_layer_k <= 0:
                    threshold_val = None
                    topk_indices = torch.empty(0, dtype=torch.long)
                else:
                    kth = max(1, min(target_layer_k, layer_scores.numel()-1))
                    vals, idxs = torch.topk(layer_scores, kth, largest=False)
                    threshold_val = float(vals.max().item())
                    topk_indices = idxs

                stats['per_layer_allocations'][lk] = {
                    'target_mask': int(target_layer_k),
                    'layer_size': int(layer_sizes[lk]),
                    'threshold': threshold_val,
                }

                for name, param in plist:
                    if name not in per_param_scores:
                        continue
                    weight = param.data
                    importance_scores = per_param_scores[name].to(weight.device)
                    numel = int(weight.numel())
                    score_min = float(importance_scores.min().item())
                    score_max = float(importance_scores.max().item())
                    score_mean = float(importance_scores.mean().item())
                    score_std = float(importance_scores.std().item())

                    if threshold_val is None or topk_indices.numel() == 0:
                        masked = 0
                        mask_ratio = 0.0
                    else:
                        start_offset = 0
                        for n2, p2 in plist:
                            if n2 == name:
                                break
                            if n2 in per_param_scores:
                                start_offset += per_param_scores[n2].numel()
                        end_offset = start_offset + numel
                        in_this = (topk_indices >= start_offset) & (topk_indices < end_offset)
                        local_indices = (topk_indices[in_this] - start_offset).to(dtype=torch.long)
                        local_indices = local_indices.to(weight.device)
                        local_mask = torch.zeros(numel, dtype=torch.bool, device=weight.device)
                        if local_indices.numel() > 0:
                            local_mask[local_indices] = True
                            flat_view = weight.view(-1)
                            flat_view[local_mask] = 0
                        masked = int(local_mask.sum().item())
                        mask_ratio = float(masked / numel) if numel > 0 else 0.0
                    masked_total += masked

                    stats['per_param'][name] = {
                        'numel': int(numel),
                        'masked': masked,
                        'ratio': (0.0 if numel == 0 else mask_ratio),
                        'mean_importance': score_mean,
                        'score_stats': {
                            'min': score_min,
                            'max': score_max,
                            'mean': score_mean,
                            'std': score_std,
                            'threshold': float(threshold_val) if threshold_val is not None else None,
                            'masked_ratio': mask_ratio
                        }
                    }
                    if len(stats['per_param']) <= 5:
                        logger.info(
                            f"Param {name} | layer {lk}: scores [{score_min:.4f}, {score_max:.4f}], "
                            f"mean={score_mean:.4f}, std={score_std:.4f}, "
                            f"masked={masked}/{numel} ({mask_ratio:.2%})")

        stats['masked_params'] = int(masked_total)
        stats['actual_mask_ratio'] = float(masked_total / total_candidates) if total_candidates > 0 else 0.0
        self.unstructured_stats = stats

        logger.info(f"{self.unstructured_scoring} pruning complete: masked {masked_total}/{total_candidates} "
                    f"({stats['actual_mask_ratio']:.2%}) parameters")

        try:
            validation = validate_pruning_distribution(self.model, stats)
            self.unstructured_stats['validation'] = validation
            logger.info(
                f"Validation - mixed_pruning_percentage: {validation.get('mixed_pruning_percentage', 0.0):.1f}%"
            )
        except Exception as e:
            logger.warning(f"Validation step failed: {e}")

        logger.info("\n" + "="*60)
        logger.info("DATA SOURCE USAGE STATISTICS")
        logger.info("="*60)
        total_params = sum(self.data_source_stats['domain_fisher'].values())
        if total_params > 0:
            for metric, counts in self.data_source_stats.items():
                per_weight_count = counts['per_weight_array']
                fallback_count = counts['fallback']
                per_weight_pct = per_weight_count / total_params * 100
                fallback_pct = fallback_count / total_params * 100
                logger.info(f"{metric.upper()}:")
                logger.info(f"  ✓ Per-weight arrays: {per_weight_count}/{total_params} ({per_weight_pct:.1f}%)")
                logger.info(f"  ⚠ Fallback values: {fallback_count}/{total_params} ({fallback_pct:.1f}%)")
                if fallback_count > 0:
                    logger.warning(f"  ⚠ WARNING: {fallback_count} parameters using fallback for {metric}")
                else:
                    logger.info(f"  ✓ SUCCESS: All parameters using per-weight arrays for {metric}")
        else:
            logger.warning("No parameters processed for data source tracking")
        logger.info("="*60)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def setup_device(self):
        """Setup device and move model"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            logger.info(f"Using {num_gpus} GPUs with batch size {self.batch_size} per GPU")

        logger.info(f"Model moved to device: {self.device}")

    def set_prompt(self, prompt: str):
        """Set instruction prompt for queries"""
        self.query_instruction_for_retrieval = prompt
        logger.info(f"Set instruction prompt: {prompt}")

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format query with instruction"""
        return f'Instruct: {task_description}\nQuery: {query}'

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries for retrieval task"""
        if self.query_instruction_for_retrieval is not None:
            input_texts = [
                self.get_detailed_instruct(self.query_instruction_for_retrieval, q)
                for q in queries
            ]
        else:
            input_texts = queries
        return self.encode(input_texts, **kwargs)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        """Encode corpus for retrieval task"""
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts)

    def pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None):
        """Apply pooling to get sentence embeddings"""
        return last_token_pool(last_hidden_state, attention_mask)

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = None, **kwargs) -> np.ndarray:
        """Encode sentences to embeddings"""
        self.model.eval()

        if batch_size is None:
            batch_size = self.batch_size

        all_embeddings = []

        for start_index in tqdm(
            range(0, len(sentences), batch_size),
            desc="Encoding batches",
            disable=len(sentences) < 256
        ):
            sentences_batch = sentences[start_index:start_index + batch_size]

            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            ).to(self.device)

            outputs = self.model(**inputs, return_dict=True)
            last_hidden_state = outputs.last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def count_model_zeros(model) -> Tuple[int, int, int]:
    """Count zero parameters in the model"""
    total_params = 0
    zero_params = 0

    try:
        with torch.no_grad():
            for param in model.parameters():
                numel = param.numel()
                total_params += numel
                zero_mask = (param == 0)
                zero_params += int(zero_mask.sum().item())
    except Exception as e:
        logger.warning(f"Failed to count zeros: {e}")
        return 0, 0

    return total_params, zero_params


def validate_pruning_distribution(model, stats: Dict) -> Dict:
    """Validate that pruning is distributed within parameters (not all-or-nothing)"""
    validation_results = {
        'parameters_with_mixed_pruning': 0,
        'parameters_all_pruned': 0,
        'parameters_none_pruned': 0,
        'problematic_parameters': []
    }

    if 'per_param' not in stats:
        return validation_results

    for param_name, param_stats in stats['per_param'].items():
        ratio = param_stats.get('ratio', 0.0)
        numel = param_stats.get('numel', 0)

        if numel == 0:
            continue

        if ratio == 0.0:
            validation_results['parameters_none_pruned'] += 1
        elif ratio >= 0.99:
            validation_results['parameters_all_pruned'] += 1
            validation_results['problematic_parameters'].append({
                'name': param_name,
                'issue': 'all_pruned',
                'ratio': ratio
            })
        elif 0.01 < ratio < 0.99:
            validation_results['parameters_with_mixed_pruning'] += 1
        else:
            validation_results['parameters_none_pruned'] += 1

    total_params = (validation_results['parameters_with_mixed_pruning'] +
                   validation_results['parameters_all_pruned'] +
                   validation_results['parameters_none_pruned'])

    if total_params > 0:
        validation_results['mixed_pruning_percentage'] = (
            validation_results['parameters_with_mixed_pruning'] / total_params * 100
        )

    if validation_results['parameters_all_pruned'] > 0:
        logger.warning(f"WARNING: {validation_results['parameters_all_pruned']} parameters "
                      f"have ALL weights pruned (ratio ≈ 1.0)")
        for prob_param in validation_results['problematic_parameters'][:5]:
            logger.warning(f"  - {prob_param['name']}: ratio={prob_param['ratio']:.4f}")

    if validation_results.get('mixed_pruning_percentage', 100.0) < 80:
        logger.warning(f"WARNING: Only {validation_results.get('mixed_pruning_percentage', 0.0):.1f}% "
                      f"of parameters have mixed pruning (should be >80%)")

    return validation_results


def excute_sparse_model(model_name_or_path: str,
                             output_dir: str,
                             batch_size: int = 64,
                             sparsify_layers: List[str] = None,
                             unstructured_pct: float = 0.0,
                             unstructured_scoring: str = 'dai',
                             general_fisher_scores: str = None,
                             domain_fisher_scores: str = None,
                             alignment_scores: str = None,
                             layer_budget_strategy: str = 'global',
                             score_postproc: str = 'zscore_sigmoid',
                             score_temperature: float = 1.0,
                             fusion_alpha: float = 0.0,
                             fusion_beta: float = 1.0,
                             rho_min: float = 0.05,
                             rho_max: float = 0.95,
                             beta: float = 1.0,
                             compression_weight: float = 0.5,
                             ):
    """Evaluate a model on domain-specific MTEB tasks with optional Information Bottleneck pruning"""
    logger.info(f"Evaluating model: {model_name_or_path}")
    logger.info(f"Using pruning method: {unstructured_scoring}")

    model = QwenSparseModel(
        model_name_or_path=model_name_or_path,
        normalize_embeddings=True,
        batch_size=batch_size,
        general_fisher_scores=general_fisher_scores,
        domain_fisher_scores=domain_fisher_scores,
        alignment_scores=alignment_scores,
        sparsify_layers=sparsify_layers,
        unstructured_pct=unstructured_pct,
        unstructured_scoring=unstructured_scoring,
        layer_budget_strategy=layer_budget_strategy,
        score_postproc=score_postproc,
        score_temperature=score_temperature,
        fusion_alpha=fusion_alpha,
        fusion_beta=fusion_beta,
        rho_min=rho_min,
        rho_max=rho_max,
        beta=beta,
        compression_weight=compression_weight,
    )

    model_name = model_name_or_path.split('/')[-1]

    if unstructured_pct > 0:
        model_name += f"_{unstructured_scoring}_prune_{unstructured_pct:.0%}"

    full_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(full_output_dir, exist_ok=True)

    actual_model = model.model.module if hasattr(model.model, 'module') else model.model
    total_params, zero_params = count_model_zeros(actual_model)

    config_info = {
        "model_path": model_name_or_path,
        "batch_size": batch_size,
        "sparsify_layers": sparsify_layers,
        "pruning_method": unstructured_scoring,
        "meta": {
            "unstructured_pruning": {
                "unstructured_pct": unstructured_pct,
                "unstructured_scoring": unstructured_scoring,
            },
            "score_files": {
                "general_fisher_scores": general_fisher_scores,
                "domain_fisher_scores": domain_fisher_scores,
                "alignment_scores": alignment_scores,
            },
        },
    }

    unstructured_masked = int(getattr(model, 'unstructured_stats', {}).get('masked_params', 0) if hasattr(model, 'unstructured_stats') else 0)

    config_info["meta"]["mask_accounting"] = {
        "total_params_before": total_params,
        "total_params_after": total_params - zero_params if zero_params else total_params,
        "masked_params": zero_params,
        "masked_ratio": zero_params / total_params if total_params > 0 else 0.0,
        "zero_params_counted": zero_params,
        "unstructured_masked_params_reported": unstructured_masked,
    }

    config_file = os.path.join(full_output_dir, "evaluation_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    logger.info(f"Configuration saved to: {config_file}")

    if hasattr(model, 'unstructured_stats'):
        usp_file = os.path.join(full_output_dir, "unstructured_prune_stats.json")
        with open(usp_file, 'w') as f:
            json.dump(model.unstructured_stats, f, indent=2)
        logger.info(f"Unstructured pruning stats saved to: {usp_file}")

    return model



def main():
    """Main function for FinMTEB evaluation with Information Bottleneck pruning only"""
    parser = argparse.ArgumentParser(description="Evaluate Qwen models on FinMTEB with Information Bottleneck pruning")
    parser.add_argument('--model_path', required=True, help="Path to the model")
    parser.add_argument('--output_dir', required=True, help="Output directory for results")
    parser.add_argument('--general_fisher_scores', type=str, default=None,
                       help='Path to general Fisher scores JSON')
    parser.add_argument('--domain_fisher_scores', type=str, default=None,
                       help='Path to domain Fisher scores JSON')
    parser.add_argument('--alignment_scores', type=str, default=None,
                       help='Path to alignment scores JSON')
    parser.add_argument('--sparsify_layers', nargs='+', default=['mlp'],
                       help="Layer types to sparsify. Use --show_layer_types to see options")
    parser.add_argument('--unstructured_pct', type=float, default=0.0,
                        help='Global unstructured sparsity ratio within selected layers (0.0-1.0)')
    parser.add_argument('--unstructured_scoring', type=str, default='dai',
                        choices=['dai'],
                        help='Scoring method for unstructured pruning (DAI only)')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for evaluation")
    parser.add_argument('--show_layer_types', action='store_true',
                       help="Show available layer types and exit")
    parser.add_argument('--layer_budget_strategy', type=str, default='global',
                        choices=['global', 'uniform'],
                        help='Global top-k across all params, or per-layer allocation strategy')
    parser.add_argument('--score_postproc', type=str, default='none',
                        choices=['zscore_sigmoid', 'minmax', 'rank', 'none'],
                        help='Per-tensor score normalization')
    parser.add_argument('--score_temperature', type=float, default=1.0,
                        help='Temperature for zscore_sigmoid postproc (smaller=sharper)')
    parser.add_argument('--fusion_alpha', type=float, default=0.0,
                        help='Fusion weight for domain_fisher*|w| term')
    parser.add_argument('--fusion_beta', type=float, default=1.0,
                        help='Fusion weight for advanced importance term')
    parser.add_argument('--rho_min', type=float, default=0.05,
                        help='Lower bound for per-layer ratio if using uniform strategy')
    parser.add_argument('--rho_max', type=float, default=0.95,
                        help='Upper bound for per-layer ratio if using uniform strategy')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for information bottleneck')
    parser.add_argument('--compression_weight', type=float, default=0.5,
                       help='Compression weight for information bottleneck')

    args = parser.parse_args()

    if args.show_layer_types:
        print_available_layer_types()
        return

    if args.unstructured_pct < 0 or args.unstructured_pct > 1:
        parser.error("unstructured_pct must be between 0.0 and 1.0")

    if args.unstructured_pct > 0 and not (args.general_fisher_scores or args.domain_fisher_scores):
        logger.warning("No score files provided for pruning. Using default values.")

    sparse_model = excute_sparse_model(
        model_name_or_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sparsify_layers=args.sparsify_layers,
        unstructured_pct=args.unstructured_pct,
        unstructured_scoring=args.unstructured_scoring,
        general_fisher_scores=args.general_fisher_scores,
        domain_fisher_scores=args.domain_fisher_scores,
        alignment_scores=args.alignment_scores,
        layer_budget_strategy=args.layer_budget_strategy,
        score_postproc=args.score_postproc,
        score_temperature=args.score_temperature,
        fusion_alpha=args.fusion_alpha,
        fusion_beta=args.fusion_beta,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        beta=args.beta,
        compression_weight=args.compression_weight,
    )
    
    test_text = "What is the capital of France?"
    test_embedding = sparse_model.encode(test_text)
    print(test_embedding)

    logger.info(f"Pruning completed!")


if __name__ == "__main__":
    main()


