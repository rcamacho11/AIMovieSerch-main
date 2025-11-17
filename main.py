import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import requests
import sqlite3
from urllib.parse import quote
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
nltk.download("punkt")

# ---------------------------------------------------------
# Optimize for Mac M-series (avoid segfaults)
# ---------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ---------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------
st.set_page_config(page_title="🎬 Movie Storyline Search", layout="wide")

TMDB_API_KEY = "2475813c017e111b42f69d3d5b8149c7"
DB_PATH = "movies.sqlite"

spell = SpellChecker()

# ---------------------------------------------------------
# SPELL CHECKER (fixed to preserve punctuation)
# ---------------------------------------------------------
def clean_query(text: str) -> str:
    """
    Corrects spelling for words only — keeps punctuation intact.
    Prevents the last period or commas from being removed.
    """
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    corrected = []
    for w in words:
        if re.match(r"\w+", w):  # only correct actual words
            corr = spell.correction(w)
            corrected.append(corr if corr else w)
        else:
            corrected.append(w)  # leave punctuation untouched
    return "".join(
        [" " + w if not re.match(r"[,.!?;:]", w) and i != 0 else w for i, w in enumerate(corrected)]
    ).strip()

# ---------------------------------------------------------
# LOAD DATA + STORED EMBEDDINGS
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT 
            t.Title,
            t.Primary_Title,
            ry.Release_Year,
            g.Genre,
            r.IMDb_Rating,
            n.Number_of_Ratings,
            s.Synopsis
        FROM title t
        LEFT JOIN release_year ry ON t.Title = ry.Title
        LEFT JOIN genre g ON t.Title = g.Title
        LEFT JOIN rating r ON t.Title = r.Title
        LEFT JOIN num_ratings n ON t.Title = n.Title
        LEFT JOIN synopsis s ON t.Title = s.Title
        GROUP BY t.Title
        ORDER BY ry.Release_Year DESC;
    """
    metadata = pd.read_sql_query(query, conn)

    vectors_df = pd.read_sql_query("SELECT * FROM storyline_vector;", conn)
    conn.close()

    titles = vectors_df["Title"].tolist()
    vectors = vectors_df.drop(columns=["Title"]).to_numpy(dtype=np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)

    return metadata, titles, vectors

# ---------------------------------------------------------
# TMDB POSTER FETCH
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_poster_tmdb(title):
    try:
        query = quote(str(title))
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=en-US"
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            data = res.json()
            if data.get("results"):
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

# ---------------------------------------------------------
# LOAD FIXED GTE MODEL
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_gte_model():
    model_name = "Alibaba-NLP/gte-base-en-v1.5"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    st.info(f"📘 Using fixed model: {model_name} on {device.upper()}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return model

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
imdb, title_list, EMBEDDINGS = load_data_from_db()
MODEL = load_gte_model()

st.title("🎬 AI Movie Search (GTE-Base-EN Only)")
st.write("Find similar movies by storyline — using fixed Alibaba GTE embeddings (768-D).")

query = st.text_area(
    "Enter your movie storyline:",
    placeholder="Example: A young man stranded at sea befriends a tiger...",
    height=100
)
top_k = st.slider("Number of results", 1, 50, 25, step=1)

# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a storyline.")
    else:
        corrected = clean_query(query)
        if corrected.lower() != query.lower():
            st.info(f"💡 Did you mean: `{corrected}`")

        with st.spinner("Searching similar storylines..."):
            q_vec = MODEL.encode([corrected], normalize_embeddings=True)
            sims = cosine_similarity(q_vec, EMBEDDINGS)[0]
            top_idx = np.argsort(-sims)[:top_k]
            top_scores = sims[top_idx]

        st.subheader("🔍 Search Results")

        num_cols = 5
        for i in range(0, len(top_idx), num_cols):
            cols = st.columns(num_cols)
            for col, idx in zip(cols, top_idx[i:i + num_cols]):
                row = imdb.iloc[idx]
                title = row.get("Primary_Title", "Unknown Title")
                year = row.get("Release_Year", "")
                synopsis = row.get("Synopsis", "No synopsis available.")
                genre = row.get("Genre", "N/A")
                rating = row.get("IMDb_Rating", None)
                num_ratings = row.get("Number_of_Ratings", None)
                similarity = int(top_scores[np.where(top_idx == idx)[0][0]] * 100)
                poster_url = get_poster_tmdb(title)

                with col:
                    st.image(poster_url, caption=f"{title} ({year})", use_container_width=True)
                    with st.expander("More info"):
                        st.markdown(f"**⭐ IMDb Rating:** {round(rating, 2) if rating else 'N/A'}")
                        st.markdown(f"**👥 Ratings Count:** {int(num_ratings) if num_ratings else 'N/A'}")
                        st.markdown(f"**🧮 Similarity:** {similarity}%")
                        st.markdown(f"**🎭 Genre:** {genre}")
                        st.markdown(f"**📖 Synopsis:** {synopsis}")

st.caption("CSE111 — Richard Camacho, Akshaya Natarajan & Ailisha Shukla · Fixed Alibaba GTE Embeddings")
