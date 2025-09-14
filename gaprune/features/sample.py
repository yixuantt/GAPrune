import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc
import os
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data: List[Dict], filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract embeddings from the last token position."""
    seq_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.size(0)), seq_lengths]


def generate_embeddings_batch(model, tokenizer, texts: List[str], 
                            max_length: int = 512, batch_size: int = 32,
                            device: Optional[str] = None) -> np.ndarray:
    """Generate embeddings for a batch of texts with memory optimization."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model
    model.eval()
    
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        
        # Tokenize with memory efficiency
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        
        # Generate embeddings without gradient computation
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state, 
                inputs.attention_mask
            )
            
            # Move to CPU and convert to numpy immediately
            embeddings.append(batch_embeddings.cpu().numpy())
        
        # Clear GPU memory
        del inputs, outputs, batch_embeddings
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    return np.vstack(embeddings)


def generate_embeddings_streaming(model_name: str, texts: List[str], 
                                max_length: int = 512, batch_size: int = 32,
                                save_interval: int = 10000) -> np.ndarray:
    """Generate embeddings with streaming to disk for very large datasets."""
    logger.info(f"Loading model: {model_name}")
    
    # Load model with memory mapping
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16,  # Use half precisio
    )
    
    # Create temporary directory for embedding chunks
    temp_dir = "temp_embeddings"
    os.makedirs(temp_dir, exist_ok=True)
    
    chunk_files = []
    current_chunk = []
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = generate_embeddings_batch(
                model, tokenizer, batch, max_length, 1, 'cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            current_chunk.append(batch_embeddings)
            
            # Save chunk to disk when reaching save_interval
            if len(current_chunk) * batch_size >= save_interval or i + batch_size >= len(texts):
                chunk_array = np.vstack(current_chunk)
                chunk_file = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.npy")
                np.save(chunk_file, chunk_array)
                chunk_files.append(chunk_file)
                
                current_chunk = []
                gc.collect()
        
        # Load and concatenate all chunks
        logger.info("Loading embedding chunks from disk...")
        all_embeddings = []
        for chunk_file in chunk_files:
            all_embeddings.append(np.load(chunk_file))
        
        final_embeddings = np.vstack(all_embeddings)
        
        # Clean up temporary files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        os.rmdir(temp_dir)
        
        return final_embeddings
        
    except Exception as e:
        # Clean up on error
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        raise e
    
    finally:
        # Clear model from memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def kmeans_diversity_sampling(embeddings: np.ndarray, metadata: List[Dict], 
                            target_size: int = 5000, niter: int = 20,
                            use_gpu: bool = True) -> List[Dict]:
    """Perform KMeans clustering for diversity sampling."""
    dim = embeddings.shape[1]
    embeddings = embeddings.astype(np.float32)
    
    if len(embeddings) <= target_size:
        return metadata[:len(embeddings)]
    
    logger.info(f"Running FAISS KMeans with target size: {target_size}")
    
    # Configure KMeans based on available resources
    if use_gpu and faiss.get_num_gpus() > 0:
        # Use GPU if available
        kmeans = faiss.Kmeans(d=dim, k=target_size, niter=niter, verbose=True, gpu=True)
    else:
        # Use CPU with optimized settings
        kmeans = faiss.Kmeans(
            d=dim, 
            k=target_size, 
            niter=niter, 
            verbose=True, 
            gpu=False,
            nredo=1,  # Reduce number of redos
            max_points_per_centroid=1000  # Limit points per centroid
        )
    
    # Train KMeans with subset if dataset is very large
    if len(embeddings) > 100000:
        logger.info("Using subset for KMeans training due to large dataset")
        subset_indices = np.random.choice(len(embeddings), 100000, replace=False)
        kmeans.train(embeddings[subset_indices])
    else:
        kmeans.train(embeddings)
    
    # Find nearest data point to each centroid
    # Use batch search to reduce memory usage
    batch_size = 1000
    selected_indices = []
    
    for i in range(0, target_size, batch_size):
        end_idx = min(i + batch_size, target_size)
        batch_centroids = kmeans.centroids[i:end_idx]
        
        # Create temporary index for batch
        if use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.IndexFlatL2(dim)
            index = faiss.index_cpu_to_all_gpus(index)
        else:
            index = faiss.IndexFlatL2(dim)
        
        index.add(embeddings)
        _, indices = index.search(batch_centroids, 1)
        selected_indices.extend(indices.flatten())
        
        del index
        gc.collect()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    
    return [metadata[i] for i in unique_indices[:target_size]]

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Diversity sampling")
    parser.add_argument("--input_file", type=str, default="data/chem-example.json")
    parser.add_argument("--output_file", type=str, default="data/sample/chem-example-output.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--target_sample_size", type=int, default=5000)
    parser.add_argument("--text_key", type=str, default="query")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_streaming", type=bool, default=True)
    return parser.parse_args()

def main():
    args = get_args()
    # ======== Configuration ========
    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name
    target_sample_size = args.target_sample_size
    text_key = args.text_key
    
    # Optimized parameters
    batch_size = args.batch_size  # Reduced from 64
    max_length = args.max_length
    use_streaming = args.use_streaming  # Enable for very large datasets
    
    # ======== Load Data ========
    logger.info(f"Loading data from {input_file}")
    data = load_data(input_file)
    
    if len(data) < target_sample_size:
        logger.warning(f"Data size ({len(data)}) is less than target sample size ({target_sample_size})")
        save_data(data, output_file)
        return
    
    # Random sampling if dataset is too large
    if len(data) > 50000:
        logger.info(f"Dataset too large ({len(data)}), sampling to 50000")
        import random
        random.seed(42)  # For reproducibility
        data = random.sample(data, 50000)
    
    texts = [item[text_key] for item in data]
    
    # ======== Generate Embeddings ========
    logger.info("Generating embeddings...")
    
    if use_streaming and len(texts) > 50000:
        embeddings = generate_embeddings_streaming(
            model_name, texts, max_length, batch_size
        )
    else:
        # Load model once
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModel.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float16,
        )
        
        embeddings = generate_embeddings_batch(
            model, tokenizer, texts, max_length, batch_size
        )
        
        # Clean up model
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # ======== KMeans Sampling ========
    logger.info("Performing KMeans diversity sampling...")
    sampled_data = kmeans_diversity_sampling(
        embeddings, data, target_sample_size,
        use_gpu=torch.cuda.is_available()
    )
    
    # ======== Save Sampled Data ========
    save_data(sampled_data, output_file)
    logger.info(f"Saved {len(sampled_data)} samples to {output_file}")


if __name__ == "__main__":
    main()