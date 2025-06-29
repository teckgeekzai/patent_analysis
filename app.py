# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import contextlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

st.set_page_config(page_title="Patent Analysis Toolkit", layout="wide")
st.title("🧠 Patent Analysis Toolkit")

# Load model safely (suppress errors/warnings)
with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
    model = SentenceTransformer('all-MiniLM-L6-v2')

# Inputs
query = st.text_area("🔍 Enter a query (optional)", "")
uploaded_file = st.file_uploader("📂 Upload CSV (must include 'title', 'abstract', optional 'claims')", type="csv")

# Run button
if st.button("⚙️ Run Analysis"):
    if not query and not uploaded_file:
        st.warning("Please provide at least a query or upload a CSV.")
    else:
        df = pd.DataFrame()

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file).fillna("")
                if not {'title', 'abstract'}.issubset(df.columns):
                    st.error("CSV must include at least 'title' and 'abstract' columns.")
                    st.stop()
                if 'claims' not in df.columns:
                    df['claims'] = ""
                df['text'] = df['title'] + ". " + df['abstract'] + ". " + df['claims']
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                st.stop()

            # TF-IDF Lexical Similarity
            tfidf = TfidfVectorizer(stop_words='english')
            X = tfidf.fit_transform(df['text'])
            lexical_similarity_matrix = cosine_similarity(X)

            # Semantic Similarity
            embeddings = model.encode(df['text'].tolist(), convert_to_tensor=False)
            semantic_similarity_matrix = cosine_similarity(embeddings)

            # Claim Length + Semantic Novelty
            df['claim_length'] = df['claims'].apply(lambda x: len(str(x).split()))
            df['semantic_novelty'] = [1 - np.mean(row) for row in semantic_similarity_matrix]

        if query:
            query_embedding = model.encode([query])[0]

            if not df.empty:
                df['relevance_score'] = [
                    np.dot(query_embedding, e) / (norm(query_embedding) * norm(e)) for e in embeddings
                ]
            else:
                # If only query is provided, analyze it against nothing
                st.warning("You provided only a query. Please upload CSV for comparative analysis.")
                st.stop()

        if not df.empty:
            # Market Value Score: weighted average
            df['market_value_score'] = (
                0.4 * df.get('relevance_score', 0).fillna(0) +
                0.3 * (df['claim_length'] / df['claim_length'].max()) +
                0.3 * df['semantic_novelty']
            )

            if 'relevance_score' in df.columns:
                st.subheader("🔝 Top 5 Relevant Patents to Your Query")
                st.dataframe(df[['title', 'relevance_score']].sort_values(by='relevance_score', ascending=False).head(5))

            st.subheader("💡 Top 5 Patents by Estimated Market Value")
            st.dataframe(df[['title', 'market_value_score']].sort_values(by='market_value_score', ascending=False).head(5))

            # Download analyzed results
            st.download_button("📥 Download Full CSV", df.to_csv(index=False), file_name="patent_analysis_results.csv")
