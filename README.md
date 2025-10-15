### 🧠 Building an AI Agent using Agentic AI (Offline – SQuAD v2 QA)

## 📘 Overview

This project builds a local, fully offline AI Agent that can answer questions intelligently using the SQuAD v2 dataset as its knowledge base.

The system combines:

  - Retriever (Sentence-Transformers + FAISS) → finds the most relevant passages.
  
  - Reader (Hugging Face QA Model – deepset/roberta-base-squad2) → extracts the best possible answer from those passages.
  
  - Streamlit UI → interactive question-answer app.
  
It runs completely offline, without calling any cloud APIs or external services.

## ⚙️ Features

✅ Offline RAG-like architecture (Retriever + Reader)

✅ FAISS vector database built locally

✅ Hugging Face models loaded from local cache

✅ Streamlit interface for interactive querying

✅ Adjustable passage retrieval (top_k)

✅ Provenance-aware responses (answers backed by SQuAD data)

## 🧱 Setup Instructions

# 1️⃣ Create Project Folder

mkdir "Building an AI Agent using Agentic AI"

cd "Building an AI Agent using Agentic AI"

# 2️⃣ Install Dependencies

💡 Works best with Python 3.11

pip install torch transformers datasets sentence-transformers faiss-cpu streamlit

## 🧮 Step 1: Build Local Knowledge Index

Run this to load SQuAD v2, embed all passages using all-MiniLM-L6-v2, and save them locally.

python build_index.py

This will:

  - Download and preprocess SQuAD v2
  
  - Chunk and embed ~24k passages
  
  - Save FAISS index and passage metadata to /models

📂 Output example:

Saved FAISS index → models/faiss.index

Saved passages → models/passages.json

Embedding dimension: 384

Passages indexed: 24678


## 🧭 Step 2: Run the Agent

Start the Streamlit interface:

streamlit run app.py

Then open your browser at:

👉 http://localhost:8501

## 💬 Step 3: Ask Questions

You can now ask factual questions (based on SQuAD/Wikipedia content):

Examples:

  - “Who discovered gravity?”
  
  - “What is the capital of India?”
  
  - “When did World War II end?”
  
  - “What is photosynthesis?”
  
  - “Who painted the Mona Lisa?”

💡 The model retrieves top-5 passages and uses RoBERTa (SQuAD2) to answer.

## 🧠 How It Works Internally

  1. Retriever (FAISS + MiniLM):
  
    - Encodes all SQuAD passages into dense vectors.
    
    - Retrieves most relevant ones based on cosine similarity.
  
  2. Reader (RoBERTa QA):
  
    - Reads the retrieved text chunks.
    
    - Extracts the most probable answer span.
  
  3. Streamlit Agent UI:
  
    - Lets you ask queries.
    
    - Displays top answers and their sources.

## 🧩 Example Interaction

Question: What is the capital of India?

Agent Answer: During British rule, the capital was Calcutta. Later, it was moved to Delhi.

(Sourced from SQuAD passage on British India history)


<img width="1366" height="596" alt="Screenshot (33)" src="https://github.com/user-attachments/assets/60b6f84c-6bb4-4e5f-9811-d5cacb18b53b" />

<img width="1366" height="597" alt="Screenshot (34)" src="https://github.com/user-attachments/assets/7bec093a-24a1-4fd9-80c6-4f5eb2887819" />

<img width="1366" height="602" alt="Screenshot (35)" src="https://github.com/user-attachments/assets/e9283196-539e-4037-90cf-3563da3ed896" />

<img width="1366" height="601" alt="Screenshot (36)" src="https://github.com/user-attachments/assets/59958e75-8ddf-41a8-99a4-264b41249977" />

<img width="1366" height="592" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/a879b1bd-5e21-4916-9588-ba66b65fc352" />

## ⚡ Performance Notes

  - Indexing ~24k passages → ~20 minutes (CPU)
  
  - Memory usage: ~2–3 GB during indexing
  
  - Query response time: ~2–4 seconds per question
  
  - No GPU required (runs entirely on CPU)

## 🛡️ Safety & Offline Mode

✅ No internet calls once models/datasets are cached

✅ Safe for local/offline academic or research use

🚫 No OpenAI, LangChain, or external APIs used

## 📚 Tech Stack

| Component | Tool                                       |
| --------- | ------------------------------------------ |
| Language  | Python 3.11                                |
| Embedding | SentenceTransformer (`all-MiniLM-L6-v2`)   |
| Retriever | FAISS                                      |
| Reader    | Hugging Face `deepset/roberta-base-squad2` |
| UI        | Streamlit                                  |
| Dataset   | SQuAD v2 (Hugging Face)                    |


## 🏁 Example Output (Console)

1. Unique contexts collected: 20233

2. Embedding dimension: 384, passages: 24678

3. Saved FAISS index → models/faiss.index

4. Streamlit running on → http://localhost:8501


### Author

Author: Ittyavira C Abraham 

MCA AI Student at Amrita Vishwa Vidyapeetham (Amrita Ahead) 
