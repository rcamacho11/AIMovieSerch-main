import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
import sqlite3
import faiss
import joblib
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from spellchecker import SpellChecker

st.set_page_config(page_title="🎬 Movie Storyline Search", layout="wide")

TMDB_API_KEY = "2475813c017e111b42f69d3d5b8149c7"
DB_PATH = "movies.db"

INDEX_PATH = "faiss_index.bin"
VEC_PATH = "tfidf_vectorizer.pkl"
VEC_ARRAY_PATH = "storyline_vectors.npy"

spell = SpellChecker()

def clean_query(text: str) -> str:
    """Fix simple spelling errors."""
    words = text.split()
    corrected = [spell.correction(w) if spell.correction(w) else w for w in words]
    return " ".join(corrected)

# LOAD DATA FROM SQLITE
@st.cache_data(show_spinner=True)
def load_data_from_db():
    if not os.path.exists(DB_PATH):
        st.error(f"Database file not found: {DB_PATH}")
        st.stop()

    conn = sqlite3.connect(DB_PATH)

    # --- Load IMDb info ---
    imdb = pd.read_sql("""
        SELECT 
            "Primary Title" AS primary_title,
            "Release Year" AS release_year,
            "Genre(s)" AS genres,
            Title AS title
        FROM imdb
    """, conn)

    # --- Load ratings (optional) ---
    try:
        ratings = pd.read_sql("SELECT rating FROM ratings", conn)
    except Exception:
        ratings = None

    conn.close()

    # Define column references (no autodetect)
    TITLE_COL = "primary_title"
    SYN_COL = "title"        # using Title as fallback storyline/description
    ID_COL = "rowid"         # implicit SQLite row id for indexing
    GENRES_COL = "genres"
    YEAR_COL = "release_year"

    return imdb, ID_COL, TITLE_COL, SYN_COL, GENRES_COL, YEAR_COL, ratings


# Call once
imdb, ID_COL, TITLE_COL, SYN_COL, GENRES_COL, YEAR_COL, RATINGS = load_data_from_db()

# TMDB POSTER FETCH
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

# LOAD OR BUILD FAISS INDEX (silent, cached)
@st.cache_resource(show_spinner=False)
def load_faiss_index(df, title_col, syn_col):
    if os.path.exists(INDEX_PATH) and os.path.exists(VEC_PATH) and os.path.exists(VEC_ARRAY_PATH):
        index = faiss.read_index(INDEX_PATH)
        vectorizer = joblib.load(VEC_PATH)
        X_norm = np.load(VEC_ARRAY_PATH)
        return vectorizer, index, X_norm
    else:
        # Build silently, no Streamlit messages
        text = (df[title_col].fillna("") + " " + df[syn_col].fillna("")).tolist()
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=200_000,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(text).astype(np.float32)
        X_norm = normalize(X, norm="l2").toarray()
        d = X_norm.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(X_norm)
        faiss.write_index(index, INDEX_PATH)
        joblib.dump(vectorizer, VEC_PATH)
        np.save(VEC_ARRAY_PATH, X_norm)
        return vectorizer, index, X_norm

VEC, INDEX, X_NORM = load_faiss_index(imdb, TITLE_COL, SYN_COL)

# UI
st.title("🎬 AI Movie Search")
st.write("Powered by FAISS")

query = st.text_area(
    "Enter your movie storyline:",
    placeholder="Example: A young man stranded at sea befriends a tiger...",
    height=100
)
top_k = st.slider("Number of results", 1, 50, 25, step=1)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a storyline.")
    else:
        corrected_query = clean_query(query)
        if corrected_query.strip().lower() != query.strip().lower():
            st.info(f"💡 Did you mean: `{corrected_query}`")
        else:
            corrected_query = query

        with st.spinner("Searching with FAISS..."):
            q_vec = VEC.transform([corrected_query]).astype(np.float32)
            q_vec = normalize(q_vec, norm="l2").toarray()
            D, I = INDEX.search(q_vec, top_k)
            top_idx = I[0]
            sims = D[0]

        st.subheader("Search Results")

        num_cols = 5
        for i in range(0, len(top_idx), num_cols):
            cols = st.columns(num_cols)
            for col, idx in zip(cols, top_idx[i:i + num_cols]):
                row = imdb.iloc[idx]
                title = row.get(TITLE_COL, "Unknown Title")
                year = row.get(YEAR_COL, "")
                synopsis = row.get(SYN_COL, "No synopsis available.")
                genres = row.get(GENRES_COL, "N/A")
                imdb_id = row.get(ID_COL, "")

                # ratings (safe)
                if RATINGS is not None and idx < len(RATINGS):
                    try:
                        rating = round(float(RATINGS.iloc[idx, 0]), 2)
                    except Exception:
                        rating = None
                else:
                    rating = None

                similarity = int(float(sims[np.where(top_idx == idx)[0][0]]) * 100)
                poster_url = get_poster_tmdb(title)
                imdb_link = f"https://www.imdb.com/title/{imdb_id}" if str(imdb_id).startswith("tt") else None

                with col:
                    st.image(poster_url, caption=f"{title} ({year})", use_column_width=True)
                    with st.expander("More info"):
                        st.markdown(f"**⭐ IMDb Rating:** {rating if rating else 'N/A'}")
                        st.markdown(f"**🧮 Similarity:** {similarity}%")
                        st.markdown(f"**🎭 Genres:** {genres}")
                        st.markdown(f"**📖 Synopsis:** {synopsis}")
                        if imdb_link:
                            st.markdown(f"[View on IMDb]({imdb_link})")

st.caption("CSE111: Richard Camacho, Akshaya Natarajan, and Ailisha Shukla")
