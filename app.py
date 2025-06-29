# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🧠 Patent Analysis Toolkit")

# Query Input
query = st.text_area("🔍 Enter a query (optional)", "")

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV (title, abstract, claims)", type="csv")

if st.button("Run Analysis"):
    if not uploaded_file and not query:
        st.warning("Please provide at least a CSV or a query.")
    else:
        if uploaded_file:
            df = pd.read_csv(uploaded_file).fillna("")
            df['text'] = df['title'] + ". " + df['abstract'] + ". " + df.get('claims', "")
            tfidf = TfidfVectorizer(stop_words='english')
            X = tfidf.fit_transform(df['text'])
            lexical_similarity_matrix = cosine_similarity(X)

            embeddings = model.encode(df['text'].tolist(), convert_to_tensor=False)
            semantic_similarity_matrix = cosine_similarity(embeddings)

            df['claim_length'] = df.get('claims', "").apply(lambda x: len(str(x).split()))
            df['semantic_novelty'] = [1 - np.mean(row) for row in semantic_similarity_matrix]
        else:
            df = pd.DataFrame()

        if query:
            query_embedding = model.encode([query])[0]
            if not df.empty:
                df['relevance_score'] = [
                    np.dot(query_embedding, e) / (norm(query_embedding) * norm(e)) for e in embeddings
                ]

        if not df.empty:
            df['market_value_score'] = (
                0.4 * df.get('relevance_score', 0) +
                0.3 * (df['claim_length'] / df['claim_length'].max()) +
                0.3 * df['semantic_novelty']
            )

            if query:
                st.subheader("🔝 Top 5 Relevant Patents")
                st.dataframe(df[['title', 'relevance_score']].sort_values(by='relevance_score', ascending=False).head(5))

            st.subheader("💡 Top 5 High Market Value Patents")
            st.dataframe(df[['title', 'market_value_score']].sort_values(by='market_value_score', ascending=False).head(5))

            # Download Button
            st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="patent_analysis.csv")

