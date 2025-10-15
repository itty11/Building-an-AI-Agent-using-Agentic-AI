import streamlit as st
from utils import Retriever, make_reader
import os
import pandas as pd
from typing import List

st.set_page_config(page_title="Local Agentic QA (SQuAD2)", layout="wide")
st.title("🔒 Local Agentic QA — SQuAD v2 (Retrieval + Reader)")

MODEL_DIR = "models"

# Sidebar: load models
with st.sidebar:
    st.header("Index & Models")
    st.write("Make sure you ran `python build_index.py` first.")
    load_btn = st.button("Load models", key="load_models")

# global cache
@st.cache_resource
def load_components():
    retriever = Retriever()
    reader = make_reader()
    return retriever, reader

if load_btn:
    try:
        retriever, reader = load_components()
        st.sidebar.success("Loaded retriever and reader.")
        st.sidebar.write(f"Passages: {len(retriever.passages)}")
    except Exception as e:
        st.sidebar.error(str(e))

# Main UI
st.markdown("""
Enter a question and the agent will:
1. Retrieve top-k relevant passages from the SQuAD-derived index.  
2. Run a QA reader over the retrieved passages.  
3. Show best answer + sources and a short provenance table.
""")

question = st.text_input("Ask a question about general QA knowledge (SQuAD contexts):", "")
top_k = st.slider("Number of passages to retrieve (top_k):", min_value=1, max_value=10, value=5, step=1)

col1, col2 = st.columns([2,1])

with col1:
    if st.button("Run Agent", key="run_agent"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                retriever, reader = load_components()
            except Exception as e:
                st.error("Models not loaded. Click 'Load models' on the left first.\n\n" + str(e))
                st.stop()

            with st.spinner("Retrieving passages..."):
                results = retriever.query(question, top_k=top_k)

            if not results:
                st.info("No passages found.")
            else:
                st.subheader("Retrieved passages (scored)")
                rows = []
                for r in results:
                    rows.append((r["idx"], r["score"], r["passage"][:300]+"..."))
                df = pd.DataFrame(rows, columns=["passage_idx","score","text_preview"])
                st.table(df)

                # Run reader over each passage and keep best
                answers = []
                for r in results:
                    context = r["passage"]
                    try:
                        out = reader(question=question, context=context)
                        ans_text = out.get("answer","")
                        score = float(out.get("score",0.0))
                    except Exception as e:
                        ans_text = ""
                        score = 0.0
                    answers.append({
                        "passage_idx": r["idx"],
                        "retrieval_score": r["score"],
                        "answer": ans_text,
                        "reader_score": score,
                        "context": context
                    })

                # rank by reader_score * retrieval_score (simple)
                for a in answers:
                    a["combined"] = a["reader_score"] * (a["retrieval_score"]+1e-6)

                answers_sorted = sorted(answers, key=lambda x: x["combined"], reverse=True)

                st.subheader("Top Answers")
                for i,a in enumerate(answers_sorted[:5]):
                    st.markdown(f"**Answer #{i+1} (combined={a['combined']:.4f}, reader={a['reader_score']:.4f}, retrieval={a['retrieval_score']:.4f})**")
                    st.write(a["answer"])
                    with st.expander("Show context passage"):
                        st.write(a["context"])

                # provenance table
                prov = pd.DataFrame([{
                    "passage_idx": a["passage_idx"],
                    "reader_score": a["reader_score"],
                    "retrieval_score": a["retrieval_score"],
                    "combined": a["combined"],
                    "answer": a["answer"][:200]
                } for a in answers_sorted])
                st.subheader("Provenance & ranking")
                st.dataframe(prov)

with col2:
    st.info("Quick tips")
    st.write("""
    • This system uses a **retriever** (sentence-transformers + FAISS) to get candidate passages,\n
    • then a **reader** (Hugging Face QA model) extracts exact answers from each passage.\n
    • Run `python build_index.py` first (creates models/faiss.index and models/passages.json).
    """)
    st.caption("Everything runs locally — no cloud APIs.\nFirst run downloads models/dataset to your cache.")
