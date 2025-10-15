import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

MODEL_DIR = "models"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
READER_MODEL = "deepset/roberta-base-squad2"  # offline HF model for squad2

class Retriever:
    def __init__(self, index_path=None, passages_path=None, emb_model_name=EMB_MODEL_NAME):
        self.index_path = index_path or os.path.join(MODEL_DIR, "faiss.index")
        self.passages_path = passages_path or os.path.join(MODEL_DIR, "passages.json")
        self.emb_model = SentenceTransformer(emb_model_name)
        self._load_index_and_passages()

    def _load_index_and_passages(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.passages_path):
            raise FileNotFoundError("Index or passages not found. Run build_index.py first.")
        self.index = faiss.read_index(self.index_path)
        with open(self.passages_path, "r", encoding="utf8") as f:
            obj = json.load(f)
        self.passages = obj["passages"]

    def query(self, text, top_k=5):
        emb = self.emb_model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.passages):
                continue
            results.append({"passage": self.passages[idx], "score": float(score), "idx": int(idx)})
        return results

def make_reader(model_name=READER_MODEL):
    # use the HF pipeline (question-answering)
    qa = pipeline("question-answering", model=model_name, tokenizer=model_name, device=-1)  # CPU
    return qa
