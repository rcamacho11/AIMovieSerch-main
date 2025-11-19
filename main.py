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

# ---------------------------------------------------------
# NLTK / ENV / TORCH SETUP
# ---------------------------------------------------------
try:
    nltk.download("punkt", quiet=True)
except:
    pass

# Optimize for Mac M-series (avoid segfaults)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="🎬 Movie Storyline Search", layout="wide")

TMDB_API_KEY = "2475813c017e111b42f69d3d5b8149c7"
DB_PATH = "movies.sqlite"

spell = SpellChecker()

# ---------------------------------------------------------
# HELPER: SPELL CHECKER
# ---------------------------------------------------------
def clean_query(text: str) -> str:
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    corrected = []
    for w in words:
        if re.match(r"\w+", w):
            corr = spell.correction(w)
            corrected.append(corr if corr else w)
        else:
            corrected.append(w)
    return "".join(
        [" " + w if not re.match(r"[,.!?;:]", w) and i != 0 else w for i, w in enumerate(corrected)]
    ).strip()


# ---------------------------------------------------------
# HELPER: DB ACCESS
# ---------------------------------------------------------
def run_sql(query: str, params=None) -> pd.DataFrame:
    """Run a SQL query against movies.sqlite and return a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


@st.cache_data(show_spinner=True)
def load_data_from_db():
    """Load metadata and embeddings from the database."""
    conn = sqlite3.connect(DB_PATH)

    # Base metadata assembled from multiple tables
    #1.
    title_df = pd.read_sql_query("SELECT Title, Primary_Title FROM title;", conn)
    #2.
    year_df = pd.read_sql_query("SELECT Title, Release_Year FROM release_year;", conn)
    #3.
    genre_df = pd.read_sql_query("SELECT Title, Genre FROM genre;", conn)
    #4.
    rating_df = pd.read_sql_query("SELECT Title, IMDb_Rating FROM rating;", conn)
    #5.
    num_df = pd.read_sql_query("SELECT Title, Number_of_Ratings FROM num_ratings;", conn)
    #6.
    synopsis_df = pd.read_sql_query("SELECT Title, Synopsis FROM synopsis;", conn)

    # Optional tables (category, character) – may not have entries for every title
    try:
        category_df = pd.read_sql_query(
            #7.
            "SELECT Title, Category AS category FROM category;", conn
        )
    except Exception:
        category_df = pd.DataFrame(columns=["Title", "category"])

    try:
        char_df = pd.read_sql_query(
            #8.
            "SELECT Title, Character AS character FROM character;", conn
        )
    except Exception:
        char_df = pd.DataFrame(columns=["Title", "character"])

    metadata = (
        title_df
        .merge(year_df, on="Title", how="left")
        .merge(genre_df, on="Title", how="left")
        .merge(rating_df, on="Title", how="left")
        .merge(num_df, on="Title", how="left")
        .merge(synopsis_df, on="Title", how="left")
        .merge(category_df, on="Title", how="left")
        .merge(char_df, on="Title", how="left")
    )

    # Load storyline vectors
    #9.
    vectors_df = pd.read_sql_query("SELECT * FROM storyline_vector;", conn)
    conn.close()

    titles = vectors_df["Title"].tolist()
    vectors = vectors_df.drop(columns=["Title"]).to_numpy(dtype=np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)

    return metadata, titles, vectors


# ---------------------------------------------------------
# HELPER: POSTER FETCH
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_poster_tmdb(title: str) -> str:
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
    # Fallback image
    return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"


# ---------------------------------------------------------
# HELPER: GTE MODEL
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_gte_model():
    model_name = "Alibaba-NLP/gte-base-en-v1.5"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    st.info(f"📘 Using model: {model_name} on {device.upper()}")
    return SentenceTransformer(model_name, trust_remote_code=True, device=device)


# ---------------------------------------------------------
# HELPER: RENDER MOVIE GRID
# ---------------------------------------------------------
def render_movie_grid(df: pd.DataFrame, similarity_scores=None, max_cols: int = 5):
    """Render a grid of movies with posters and basic info."""
    if df.empty:
        st.info("No movies found for this selection.")
        return

    num_cols = max_cols
    for i in range(0, len(df), num_cols):
        cols = st.columns(num_cols)
        for col, (_, row) in zip(cols, df.iloc[i:i + num_cols].iterrows()):
            title = row.get("Primary_Title", row.get("Title", "Unknown Title"))
            year = row.get("Release_Year", "")
            synopsis = row.get("Synopsis", "No synopsis available.")
            genre = row.get("Genre", "N/A")
            rating = row.get("IMDb_Rating", None)
            category = row.get("category", "N/A")
            character = row.get("character", "N/A")
            num_ratings = row.get("Number_of_Ratings", None)

            sim_display = None
            if similarity_scores is not None:
                sim_display = similarity_scores.get(row.get("Title"))

            poster_url = get_poster_tmdb(title)

            with col:
                st.image(
                    poster_url,
                    caption=f"{title} ({year})" if year else title,
                    use_container_width=True,
                )
                with st.expander("More info"):
                    if rating is not None:
                        st.markdown(f"**⭐ IMDb Rating:** {rating}")
                    if num_ratings is not None:
                        st.markdown(f"**👥 Ratings Count:** {num_ratings}")
                    if sim_display is not None:
                        st.markdown(f"**🧮 Similarity:** {sim_display}%")
                    st.markdown(f"**🎭 Genre:** {genre}")
                    st.markdown(f"**📂 Category:** {category}")
                    st.markdown(f"**🎙️ Character / Job:** {character}")
                    st.markdown(f"**📖 Synopsis:** {synopsis}")


# ---------------------------------------------------------
# LOAD DB + MODEL ONCE (CACHED)
# ---------------------------------------------------------
imdb, title_list, EMBEDDINGS = load_data_from_db()
MODEL = load_gte_model()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🎬 Storyline Search",
        "🎭 Browse by Genre",
        "⭐ Top Rated Movies",
        "📅 Year Explorer",
        "📈 Popular Movies (Most Ratings)",
        "➕ Add to the Catalogue",
        "🛠️ Movie Record Manager",
        "📊 Analytics & Longest Synopses",
        "🎲 Random Movie",
    ],
)


# ---------------------------------------------------------
# PAGE: STORYLINE SEARCH (MAIN FEATURE)
# ---------------------------------------------------------
if page == "🎬 Storyline Search":
    st.title("🎬 AI Movie Search (GTE-Base-EN Only)")
    st.write("Find similar movies by storyline — using 768-dim GTE embeddings from the database.")

    query = st.text_area(
        "Enter your movie storyline:",
        placeholder="Example: A young man stranded at sea befriends a tiger...",
        height=100,
    )

    top_k = st.slider("Number of results", 1, 50, 25)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a storyline.")
        else:
            corrected = clean_query(query)
            if corrected.lower() != query.lower():
                st.info(f"💡 Did you mean: `{corrected}`")

            with st.spinner("Searching..."):
                q_vec = MODEL.encode([corrected], normalize_embeddings=True)
                sims = cosine_similarity(q_vec, EMBEDDINGS)[0]
                top_idx = np.argsort(-sims)[:top_k]
                top_scores = sims[top_idx]

            st.subheader("🔍 Search Results")

            # Build a mapping of Title -> similarity %
            sim_map = {}
            for idx, score in zip(top_idx, top_scores):
                title = imdb.iloc[idx]["Title"]
                sim_map[title] = int(score * 100)

            results_df = imdb.iloc[top_idx].copy()
            render_movie_grid(results_df, similarity_scores=sim_map, max_cols=5)

    st.caption("CSE111 — Richard Camacho, Akshaya Natarajan & Ailisha Shukla · Fixed GTE Embeddings")


# ---------------------------------------------------------
# PAGE: BROWSE BY GENRE
# ---------------------------------------------------------
elif page == "🎭 Browse by Genre":
    st.title("🎭 Browse by Genre")

    # Get list of distinct genres
    genres_df = run_sql("SELECT DISTINCT Genre FROM genre WHERE Genre IS NOT NULL AND Genre <> '' ORDER BY Genre;")
    genre_list = genres_df["Genre"].tolist()

    genre = st.selectbox("Select a genre:", ["(choose)"] + genre_list)

    if genre != "(choose)":
        st.subheader(f"Movies in Genre: {genre}")
        df = run_sql(
            #10.
            """
            SELECT t.Title, t.Primary_Title, ry.Release_Year, g.Genre, r.IMDb_Rating,
                   n.Number_of_Ratings, s.Synopsis
            FROM title t
            JOIN genre g ON t.Title = g.Title
            LEFT JOIN release_year ry ON t.Title = ry.Title
            LEFT JOIN rating r ON t.Title = r.Title
            LEFT JOIN num_ratings n ON t.Title = n.Title
            LEFT JOIN synopsis s ON t.Title = s.Title
            WHERE g.Genre = ?
            ORDER BY r.IMDb_Rating DESC;
            """,
            params=(genre,),
        )
        render_movie_grid(df)


# ---------------------------------------------------------
# PAGE: TOP RATED MOVIES
# ---------------------------------------------------------
elif page == "⭐ Top Rated Movies":
    st.title("⭐ Top Rated Movies")

    min_rating = st.slider("Minimum IMDb rating", 0.0, 10.0, 8.5, 0.1)
    limit = st.slider("How many movies to show?", 5, 50, 20)

    if st.button("Show Top Rated"):
        df = run_sql(
            f"""
            #11.
            SELECT t.Title, t.Primary_Title, ry.Release_Year, g.Genre,
                   r.IMDb_Rating, n.Number_of_Ratings, s.Synopsis
            FROM title t
            JOIN rating r ON t.Title = r.Title
            LEFT JOIN release_year ry ON t.Title = ry.Title
            LEFT JOIN genre g ON t.Title = g.Title
            LEFT JOIN num_ratings n ON t.Title = n.Title
            LEFT JOIN synopsis s ON t.Title = s.Title
            WHERE r.IMDb_Rating >= ?
            ORDER BY r.IMDb_Rating DESC
            LIMIT {limit};
            """,
            params=(min_rating,),
        )
        render_movie_grid(df)


# ---------------------------------------------------------
# PAGE: YEAR EXPLORER
# ---------------------------------------------------------
elif page == "📅 Year Explorer":
    st.title("📅 Explore Movies by Release Year")

    years_df = run_sql("SELECT DISTINCT Release_Year FROM release_year WHERE Release_Year IS NOT NULL ORDER BY Release_Year;")
    years = years_df["Release_Year"].tolist()
    if years:
        year = st.selectbox("Select a year:", years)
        df = run_sql(
            #12.
            """
            SELECT t.Title, t.Primary_Title, ry.Release_Year, g.Genre, r.IMDb_Rating,
                   n.Number_of_Ratings, s.Synopsis
            FROM release_year ry
            JOIN title t ON ry.Title = t.Title
            LEFT JOIN genre g ON t.Title = g.Title
            LEFT JOIN rating r ON t.Title = r.Title
            LEFT JOIN num_ratings n ON t.Title = n.Title
            LEFT JOIN synopsis s ON t.Title = s.Title
            WHERE ry.Release_Year = ?
            ORDER BY r.IMDb_Rating DESC;
            """,
            params=(int(year),),
        )
        render_movie_grid(df)
    else:
        st.info("No release year data found in the database.")


# ---------------------------------------------------------
# PAGE: POPULAR MOVIES (MOST RATINGS)
# ---------------------------------------------------------
elif page == "📈 Popular Movies (Most Ratings)":
    st.title("📈 Most Popular Movies (by Number of Ratings)")

    limit = st.slider("How many movies to show?", 5, 50, 20)

    if st.button("Show Most Popular"):
        df = run_sql(
            #13.
            f"""
            SELECT t.Title, t.Primary_Title, ry.Release_Year, g.Genre,
                   r.IMDb_Rating, n.Number_of_Ratings, s.Synopsis
            FROM title t
            JOIN num_ratings n ON t.Title = n.Title
            LEFT JOIN rating r ON t.Title = r.Title
            LEFT JOIN release_year ry ON t.Title = ry.Title
            LEFT JOIN genre g ON t.Title = g.Title
            LEFT JOIN synopsis s ON t.Title = s.Title
            ORDER BY n.Number_of_Ratings DESC
            LIMIT {limit};
            """
        )
        render_movie_grid(df)

# ---------------------------------------------------------
# PAGE: ANALYTICS & LONGEST SYNOPSES
# ---------------------------------------------------------
elif page == "📊 Analytics & Longest Synopses":
    st.title("📊 Analytics & Longest Synopses")

    st.subheader("1. Movies with Multiple Genres")
    df_multi = run_sql(
        #14. 
        """
        SELECT Title, COUNT(*) AS GenreCount
        FROM genre
        GROUP BY Title
        HAVING COUNT(*) > 1
        ORDER BY GenreCount DESC;
        """
    )
    st.dataframe(df_multi)

    st.subheader("2. Movies Newer Than Average Release Year")
    df_newer = run_sql(
        #15.
        """
        SELECT t.Title, t.Primary_Title, ry.Release_Year
        FROM release_year ry
        JOIN title t ON ry.Title = t.Title
        WHERE ry.Release_Year >
            (SELECT AVG(Release_Year) FROM release_year WHERE Release_Year IS NOT NULL)
        ORDER BY ry.Release_Year DESC;
        """
    )
    st.dataframe(df_newer)

    st.subheader("3. Movies Above Average IMDb Rating")
    df_above_avg = run_sql(
        #16.
        """
        SELECT t.Title, t.Primary_Title, r.IMDb_Rating
        FROM rating r
        JOIN title t ON r.Title = t.Title
        WHERE r.IMDb_Rating >
            (SELECT AVG(IMDb_Rating) FROM rating WHERE IMDb_Rating IS NOT NULL)
        ORDER BY r.IMDb_Rating DESC;
        """
    )
    st.dataframe(df_above_avg)

    st.subheader("4. Top 10 Longest Synopses")
    df_long = run_sql(
        #17.
        """
        SELECT t.Title, t.Primary_Title, LENGTH(s.Synopsis) AS Synopsis_Length
        FROM synopsis s
        JOIN title t ON s.Title = t.Title
        WHERE s.Synopsis IS NOT NULL
        ORDER BY Synopsis_Length DESC
        LIMIT 10;
        """
    )
    st.dataframe(df_long)


# ---------------------------------------------------------
# PAGE: RANDOM MOVIE
# ---------------------------------------------------------
elif page == "🎲 Random Movie":
    st.title("🎲 Random Movie Generator")

    if st.button("Give me a random movie"):
        df = run_sql(
            #18.
            """
            SELECT t.Title, t.Primary_Title, ry.Release_Year, g.Genre,
                   r.IMDb_Rating, n.Number_of_Ratings, s.Synopsis
            FROM title t
            LEFT JOIN release_year ry ON t.Title = ry.Title
            LEFT JOIN genre g ON t.Title = g.Title
            LEFT JOIN rating r ON t.Title = r.Title
            LEFT JOIN num_ratings n ON t.Title = n.Title
            LEFT JOIN synopsis s ON t.Title = s.Title
            ORDER BY RANDOM()
            LIMIT 1;
            """
        )
        render_movie_grid(df, max_cols=1)
    else:
        st.info("Click the button to get a random recommendation!")
        
# ---------------------------------------------------------
# PAGE: ADD TO THE CATALOGUE
# ---------------------------------------------------------
elif page == "➕ Add to the Catalogue":
    st.title("➕ Add to the Catalogue")

    title = st.text_input("Movie Title")
    year = st.number_input("Release Year", min_value=1800, max_value=2100)
    genre = st.text_input("Genre")
    rating = st.number_input("IMDb Rating (0–10)", min_value=0.0, max_value=10.0, step=0.1)
    synopsis = st.text_area("Synopsis")

    if st.button("Add Movie"):
        if title.strip() == "":
            st.error("Title cannot be empty.")
        else:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            try:
                #19.
                cur.execute("INSERT INTO title (Title, Primary_Title) VALUES (?, ?)", (title, title))
                #20.
                cur.execute("INSERT INTO release_year (Title, Release_Year) VALUES (?, ?)", (title, year))
                #21.
                cur.execute("INSERT INTO genre (Title, Genre) VALUES (?, ?)", (title, genre))
                #22.
                cur.execute("INSERT INTO rating (Title, IMDb_Rating) VALUES (?, ?)", (title, rating))
                #23.
                cur.execute("INSERT INTO synopsis (Title, Synopsis) VALUES (?, ?)", (title, synopsis))
                conn.commit()
                st.success(f"'{title}' has been successfully added!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                conn.close()

# ---------------------------------------------------------
# PAGE: MOVIE RECORD MANAGER
# ---------------------------------------------------------
elif page == "🛠️ Movie Record Manager":
    st.title("🛠️ Movie Record Manager")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT Title FROM title ORDER BY Title ASC", conn)
    conn.close()

    if df.empty:
        st.warning("No movies found in the database.")
    else:
        titles = df["Title"].tolist()
        selected = st.selectbox("Choose a movie:", titles)

        st.subheader("Update Movie Information")
        new_title = st.text_input("New Title", value=selected)
        new_rating = st.number_input("New Rating (0–10)", min_value=0.0, max_value=10.0, step=0.1)

        if st.button("Update Movie"):
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                #24.
                cur.execute("UPDATE title SET Primary_Title=? WHERE Title=?", (new_title, selected))
                #25.
                cur.execute("UPDATE rating SET IMDb_Rating=? WHERE Title=?", (new_rating, selected))
                conn.commit()
                st.success("Movie information updated!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                conn.close()

        st.divider()

        st.subheader("Delete Movie Record")
        st.warning("⚠️ This action cannot be undone!")

        if st.button("Delete Movie"):
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                #26.
                cur.execute("DELETE FROM rating WHERE Title=?", (selected,))
                #27.
                cur.execute("DELETE FROM genre WHERE Title=?", (selected,))
                #28.
                cur.execute("DELETE FROM release_year WHERE Title=?", (selected,))
                #29.
                cur.execute("DELETE FROM synopsis WHERE Title=?", (selected,))
                #30.
                cur.execute("DELETE FROM title WHERE Title=?", (selected,))
                conn.commit()
                st.success(f"'{selected}' has been deleted.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                conn.close()