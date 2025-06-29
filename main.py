# streamlit_google_patents_scraper.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
import numpy as np
import time

st.title("🔎 Patent Search & Analysis (Google Patents)")

model = SentenceTransformer('all-MiniLM-L6-v2')

query = st.text_input("🧠 Enter your patent idea/query:")
max_results = st.slider("🔢 Number of patents to scrape", 5, 30, 10)

def fetch_google_patents(query, max_results=10):
    base_url = "https://patents.google.com/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(base_url, params=params, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    items = soup.select("search-result-item")  # old selector might not work
    if not items:
        items = soup.find_all("tr", {"data-result": True})  # fallback for list layout

    for item in items[:max_results]:
        try:
            link_tag = item.find("a", href=True)
            title = link_tag.text.strip() if link_tag else "N/A"
            link = "https://patents.google.com" + link_tag['href'] if link_tag else "N/A"
            abstract = item.find("div", class_="abstract").text.strip() if item.find("div", class_="abstract") else ""
            assignee = item.find("span", {"itemprop": "assigneeOriginal"}).text.strip() if item.find("span", {"itemprop": "assigneeOriginal"}) else "N/A"
            date = item.find("td", {"itemprop": "priorityDate"}).text.strip() if item.find("td", {"itemprop": "priorityDate"}) else "N/A"

            results.append({
                "title": title,
                "abstract": abstract,
                "assignee": assignee,
                "date": date,
                "link": link
            })
        except Exception as e:
            continue

    return pd.DataFrame(results)

if st.button("🔍 Search and Analyze"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching results from Google Patents..."):
            df = fetch_google_patents(query, max_results)
            time.sleep(1)

        if df.empty:
            st.error("No patents found or scraping failed.")
        else:
            # Fill missing data
            df['text'] = df['title'] + ". " + df['abstract']
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['text'])
            lexical_similarity_matrix = cosine_similarity(tfidf_matrix)

            embeddings = model.encode(df['text'].tolist(), convert_to_tensor=False)
            query_embedding = model.encode([query])[0]

            df['semantic_novelty'] = [1 - np.mean(row) for row in cosine_similarity(embeddings)]
            df['relevance_score'] = [
                np.dot(query_embedding, e) / (norm(query_embedding) * norm(e)) for e in embeddings
            ]
            df['claim_length'] = df['abstract'].apply(lambda x: len(x.split()))
            df['market_value_score'] = (
                0.4 * df['relevance_score'] +
                0.3 * (df['claim_length'] / df['claim_length'].max()) +
                0.3 * df['semantic_novelty']
            )

            st.success("Analysis complete!")

            st.subheader("🔝 Top 5 Relevant Patents")
            st.dataframe(df[['title', 'relevance_score', 'link']].sort_values(by='relevance_score', ascending=False).head(5))

            st.subheader("💰 Top 5 Market Value Patents")
            st.dataframe(df[['title', 'market_value_score', 'link']].sort_values(by='market_value_score', ascending=False).head(5))

            st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="patent_analysis_google.csv")
