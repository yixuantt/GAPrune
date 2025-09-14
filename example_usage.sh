#!/bin/bash

# sample data
python gaprune/features/sample.py \
    --input_file data/general-example.json \
    --output_file data/sample/general-example-output.json \
    --model_name Qwen/Qwen3-Embedding-0.6B \
    --target_sample_size 5000 \
    --text_key query \
    --batch_size 32 \
    --max_length 512 \
    --use_streaming True

# calculate fisher information
python gaprune/features/fisher.py \
    --model_name Qwen/Qwen3-Embedding-4B \
    --batch_size 4 \
    --max_samples 5000 \
    --output_dir output/fisher \
    --data_path data/sample/general-example-output.json

python gaprune/features/fisher.py \
    --model_name Qwen/Qwen3-Embedding-4B \
    --batch_size 4 \
    --max_samples 5000 \
    --output_dir output/fisher \
    --data_path data/sample/chem-example-output.json

# calculate gradient alignment
python gaprune/features/gradient.py \
    --model_name Qwen/Qwen3-Embedding-4B \
    --batch_size 4 \
    --max_samples 5000 \
    --output_dir output/gradient \
    --domain_data_path data/sample/chem-example-output.json \
    --general_data_path data/sample/general-example-output.json

# test the sparse model, please replace the alignment_scores, domain_fisher_scores, and general_fisher_scores with the actual paths
python gaprune/model/SparseModel.py \
  --model_path Qwen/Qwen3-Embedding-4B \
  --output_dir output/model\
  --batch_size 8 \
  --sparsify_layers mlp \
  --unstructured_pct 0.5 \
  --unstructured_scoring dai \
  --alignment_scores output/gradient/Qwen3-Embedding-4B/alignment_scores_Qwen3-Embedding-4B_20250914_001132.json \
  --domain_fisher_scores output/fisher/chem-example_fisher_complete_scores_20250913_235811.json \
  --general_fisher_scores output/fisher/general-example-output_fisher_complete_scores_20250913_235418.json
