import os
import json
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import argparse
import math

MODEL_DIR = "models"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast

os.makedirs(MODEL_DIR, exist_ok=True)

def chunk_text(text, max_tokens=120, sep=" "):
    # Simple chunk by words (not token-based). Keeps overlapping windows.
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks = []
    stride = max_tokens // 3
    i = 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        chunks.append(" ".join(chunk))
        if i + max_tokens >= len(words):
            break
        i += max_tokens - stride
    return chunks

def build_passages_from_squad(version="v2"):
    print("Loading SQuAD v2 dataset (may download ~100 MB)...")
    ds = load_dataset("squad_v2")
    # We will use training + validation contexts (unique)
    contexts = set()
    for split in ["train", "validation"]:
        for ex in tqdm(ds[split], desc=f"Reading {split}"):
            ctx = ex["context"].strip()
            contexts.add(ctx)
    print(f"Unique contexts collected: {len(contexts)}")
    passages = []
    meta = []
    for idx, ctx in enumerate(tqdm(list(contexts), desc="Chunking contexts")):
        chunks = chunk_text(ctx, max_tokens=150)
        for c_i, chunk in enumerate(chunks):
            # minimal cleanup
            text = " ".join([w for w in chunk.split() if w not in ENGLISH_STOP_WORDS])
            passages.append(text)
            meta.append({
                "context_id": idx,
                "chunk_id": c_i,
                "original_length": len(chunk.split())
            })
    return passages, meta

def embed_and_index(passages, emb_model_name=EMB_MODEL_NAME, index_path=os.path.join(MODEL_DIR,"faiss.index"), ids_path=os.path.join(MODEL_DIR,"passages.json"), batch_size=256):
    print("Loading embedding model:", emb_model_name)
    emb_model = SentenceTransformer(emb_model_name)
    n = len(passages)
    dim = emb_model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}, passages: {n}")

    # Create FAISS index (IndexFlatIP on normalized vectors => cosine)
    index = faiss.IndexFlatIP(dim)
    ids = []

    # process in batches
    for i in tqdm(range(0, n, batch_size), desc="Embedding batches"):
        batch = passages[i:i+batch_size]
        emb = emb_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        # normalize to unit vectors for cosine similarity via inner product
        faiss.normalize_L2(emb)
        index.add(emb)
        ids.extend(list(range(i, i+len(batch))))

    # save index
    faiss.write_index(index, index_path)
    print("Saved FAISS index to", index_path)

    # save passages with meta
    with open(ids_path, "w", encoding="utf8") as f:
        json.dump({"passages": passages}, f, ensure_ascii=False)
    print("Saved passages to", ids_path)

    return index_path, ids_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index even if models exist")
    args = parser.parse_args()

    idx_path = os.path.join(MODEL_DIR,"faiss.index")
    passages_path = os.path.join(MODEL_DIR,"passages.json")

    if not args.rebuild and os.path.exists(idx_path) and os.path.exists(passages_path):
        print("Index and passages already exist. Use --rebuild to recreate.")
        return

    passages, meta = build_passages_from_squad()
    embed_and_index(passages)

if __name__ == "__main__":
    main()
