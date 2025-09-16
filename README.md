<div align="center">
  <img src="https://github.com/yixuantt/picx-images-hosting/raw/master/logo.7lkco20x1m.webp" alt="GAPrune Logo" width="800">
  
  # GAPrune: Gradient-Alignment Pruning for Domain-Aware Embeddings
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![arXiv](https://img.shields.io/badge/arXiv-2509.10844-b31b1b.svg)](https://arxiv.org/abs/2509.10844)
</div>


GAPrune is a novel pruning framework for domain-specific pruning of embedding models. It preserves both general linguistic foundations and domain-specific capabilities for developing smaller domain embedding models.

## 🎯 Key Features

- **Domain-Aware Pruning**: Considers both domain importance and general linguistic foundation
- **Fisher Information Integration**: Uses Fisher Information to measure parameter importance
- **Gradient Alignment Analysis**: Assesses parameter behavior through general-domain gradient alignment
- **DAI Scoring**: Combines domain and alignment signals for optimal pruning decisions
- **High Performance**: Maintains within 2.5% of dense model performance at 50% sparsity in one-shot pruning.

## 🚀 Quick Start

### Prerequisites

```bash
# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Install dependencies
pip install torch transformers numpy tqdm python-dotenv
```

### Data Format

The framework expects JSON files with the following structure:

```json
[
  {
    "query": "Your query text here",
    "positive": "Positive passage for training",
    "negative": "Negative passage for training"
  }
]
```

### Basic Usage

1. **Sample your data** (optional, for large datasets):
```bash
python gaprune/features/sample.py \
    --input_file data/general-example.json \
    --output_file data/sample/general-example-output.json \
    --model_name Qwen/Qwen3-Embedding-0.6B \
    --target_sample_size 5000 \
    --text_key query \
    --batch_size 32 \
    --max_length 512 \
    --use_streaming True
```

2. **Calculate Fisher Information** for both general and domain data:
```bash
# General domain Fisher scores
python gaprune/features/fisher.py \
    --model_name Qwen/Qwen3-Embedding-4B \
    --batch_size 4 \
    --max_samples 5000 \
    --output_dir output/fisher \
    --data_path data/sample/general-example-output.json

# Domain-specific Fisher scores
python gaprune/features/fisher.py \
    --model_name Qwen/Qwen3-Embedding-4B \
    --batch_size 4 \
    --max_samples 5000 \
    --output_dir output/fisher \
    --data_path data/sample/chem-example-output.json
```

3. **Calculate Gradient Alignment**:
```bash
python gaprune/features/gradient.py \
    --model_name Qwen/Qwen3-Embedding-4B \
    --batch_size 4 \
    --max_samples 5000 \
    --output_dir output/gradient \
    --domain_data_path data/sample/chem-example-output.json \
    --general_data_path data/sample/general-example-output.json
```

4. **Apply GAPrune pruning**: please replace the alignment_scores, domain_fisher_scores, and general_fisher_scores with the actual paths
```bash
python gaprune/model/SparseModel.py \
    --model_path Qwen/Qwen3-Embedding-4B \
    --output_dir output/model \
    --batch_size 8 \
    --sparsify_layers mlp \
    --unstructured_pct 0.5 \
    --unstructured_scoring dai \
    --alignment_scores output/gradient/Qwen3-Embedding-4B/alignment_scores_Qwen3-Embedding-4B_20250914_001132.json \
    --domain_fisher_scores output/fisher/chem-example_fisher_complete_scores_20250913_235811.json \
    --general_fisher_scores output/fisher/general-example-output_fisher_complete_scores_20250913_235418.json
```

## 📁 Project Structure

```
GAPrune/
├── gaprune/
│   ├── features/
│   │   ├── fisher.py          # Fisher Information calculation
│   │   ├── gradient.py        # Gradient alignment analysis
│   │   ├── sample.py          # Data sampling utilities
│   │   └── loss_utils.py      # Loss computation utilities
│   └── model/
│       ├── SparseModel.py     # Main pruning implementation
│       └── utils.py           # Model utilities
├── data/
│   ├── general-example.json   # General domain example data
│   └── chem-example.json      # Domain-specific example data
├── output/                    # Generated outputs
│   ├── fisher/               # Fisher Information scores
│   └── gradient/             # Gradient alignment scores
└── example_usage.sh          # Complete usage example
```

## 🔧 Configuration Options

### Layer Types for Sparsification

Available layer types:
- `mlp`: MLP layers (gate_proj, up_proj, down_proj)
- `attention`: Attention layers (q_proj, k_proj, v_proj, o_proj)
- `layernorm`: Normalization layers
- `embed`: Embedding layer
- `final_norm`: Final normalization layer
- `all`: All layers except final normalization

### Pruning Parameters

- `--unstructured_pct`: Global sparsity ratio (0.0-1.0)
- `--sparsify_layers`: Layer types to sparsify
- `--layer_budget_strategy`: Allocation strategy (`global` or `uniform`)
- `--score_postproc`: Score normalization (`zscore_sigmoid`, `minmax`, `rank`, `none`)


## 📚 Citation

If you use GAPrune in your research, please cite our paper:

```bibtex
@misc{tang2025gaprunegradientalignmentpruningdomainaware,
      title={GAPrune: Gradient-Alignment Pruning for Domain-Aware Embeddings}, 
      author={Yixuan Tang and Yi Yang},
      year={2025},
      eprint={2509.10844},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.10844}, 
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For questions or support, please open an issue on GitHub.
