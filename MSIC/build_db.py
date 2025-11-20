import pandas as pd
import sqlite3
import os

# Paths
IMDB_PATH = "cleaned_imdb_data.csv"
RATINGS_PATH = "storyline_ratings.csv"      # still here if you want to use later
VECTORS_PATH = "storyline_vector.csv"
PRINCIPALS_PATH = "title.principals.tsv"    # NEW DATASET

# Create / reset database
if os.path.exists("movies.sqlite"):
    os.remove("movies.sqlite")
conn = sqlite3.connect("movies.sqlite")
cur = conn.cursor()

# Enforce foreign keys
cur.execute("PRAGMA foreign_keys = ON;")

# Drop old tables
cur.executescript("""
DROP TABLE IF EXISTS movie_job;
DROP TABLE IF EXISTS movie_character;
DROP TABLE IF EXISTS movie_category;
DROP TABLE IF EXISTS movie_genre;

DROP TABLE IF EXISTS job_ref;
DROP TABLE IF EXISTS character_ref;
DROP TABLE IF EXISTS category_ref;
DROP TABLE IF EXISTS genre_ref;

DROP TABLE IF EXISTS release_year;
DROP TABLE IF EXISTS rating;
DROP TABLE IF EXISTS num_ratings;
DROP TABLE IF EXISTS synopsis;
DROP TABLE IF EXISTS storyline_vector;
DROP TABLE IF EXISTS title;
""")

# Core tables 
cur.execute("""
CREATE TABLE title (
    Title TEXT PRIMARY KEY,
    Primary_Title TEXT
);
""")

cur.execute("""
CREATE TABLE release_year (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Release_Year INTEGER,
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE
);
""")

cur.execute("""
CREATE TABLE rating (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    IMDb_Rating REAL,
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE
);
""")

cur.execute("""
CREATE TABLE num_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Number_of_Ratings INTEGER,
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE
);
""")

cur.execute("""
CREATE TABLE synopsis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Synopsis TEXT,
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE
);
""")

# Reference tables for many-to-many
cur.execute("""
CREATE TABLE genre_ref (
    GenreID INTEGER PRIMARY KEY AUTOINCREMENT,
    GenreName TEXT UNIQUE
);
""")

cur.execute("""
CREATE TABLE category_ref (
    CategoryID INTEGER PRIMARY KEY AUTOINCREMENT,
    CategoryName TEXT UNIQUE
);
""")

cur.execute("""
CREATE TABLE character_ref (
    CharacterID INTEGER PRIMARY KEY AUTOINCREMENT,
    CharacterName TEXT UNIQUE
);
""")

cur.execute("""
CREATE TABLE job_ref (
    JobID INTEGER PRIMARY KEY AUTOINCREMENT,
    JobName TEXT UNIQUE
);
""")

# Junction tables (many-to-many)
cur.execute("""
CREATE TABLE movie_genre (
    Title TEXT,
    GenreID INTEGER,
    PRIMARY KEY (Title, GenreID),
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE,
    FOREIGN KEY (GenreID) REFERENCES genre_ref(GenreID)
);
""")

cur.execute("""
CREATE TABLE movie_category (
    Title TEXT,
    CategoryID INTEGER,
    PRIMARY KEY (Title, CategoryID),
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE,
    FOREIGN KEY (CategoryID) REFERENCES category_ref(CategoryID)
);
""")

cur.execute("""
CREATE TABLE movie_character (
    Title TEXT,
    CharacterID INTEGER,
    PRIMARY KEY (Title, CharacterID),
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE,
    FOREIGN KEY (CharacterID) REFERENCES character_ref(CharacterID)
);
""")

cur.execute("""
CREATE TABLE movie_job (
    Title TEXT,
    JobID INTEGER,
    PRIMARY KEY (Title, JobID),
    FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE,
    FOREIGN KEY (JobID) REFERENCES job_ref(JobID)
);
""")

# Helper
def safe_to_sql(df, table_name):
    if df is not None and not df.empty:
        df.to_sql(table_name, conn, if_exists="append", index=False)

print("Loading IMDb data...")
imdb_df = pd.read_csv(IMDB_PATH, low_memory=False)
imdb_df.columns = [c.strip() for c in imdb_df.columns]

#  Insert into title
# Assume first two columns are Title ID + Primary title
title_df = imdb_df.iloc[:, :2].drop_duplicates(subset=[imdb_df.columns[0]])
title_df.columns = ["Title", "Primary_Title"]
safe_to_sql(title_df, "title")

# Populate simple attribute tables
print(" Populating attribute tables...")

if "Release Year" in imdb_df.columns:
    ry_df = imdb_df[["Title", "Release Year"]].rename(columns={"Release Year": "Release_Year"})
    safe_to_sql(ry_df, "release_year")

if "IMDb Rating" in imdb_df.columns:
    rating_df = imdb_df[["Title", "IMDb Rating"]].rename(columns={"IMDb Rating": "IMDb_Rating"})
    safe_to_sql(rating_df, "rating")

if "Number of Ratings" in imdb_df.columns:
    nr_df = imdb_df[["Title", "Number of Ratings"]].rename(columns={"Number of Ratings": "Number_of_Ratings"})
    safe_to_sql(nr_df, "num_ratings")

if "Synopsis" in imdb_df.columns:
    syn_df = imdb_df[["Title", "Synopsis"]]
    safe_to_sql(syn_df, "synopsis")

# GENRE many-to-many
print(" Building Movie–Genre many-to-many...")

genre_pairs = []

if "Genre(s)" in imdb_df.columns:
    for _, row in imdb_df[["Title", "Genre(s)"]].dropna().iterrows():
        title = row["Title"]
        genres_raw = str(row["Genre(s)"])
        for g in genres_raw.split(","):
            g = g.strip()
            if g:
                genre_pairs.append({"Title": title, "GenreName": g})

if genre_pairs:
    genre_pairs_df = pd.DataFrame(genre_pairs).drop_duplicates()

    unique_genres = sorted(genre_pairs_df["GenreName"].unique())
    genre_ref_df = pd.DataFrame({
        "GenreName": unique_genres
    })
    safe_to_sql(genre_ref_df, "genre_ref")

    # Get IDs back from DB
    genre_ref_db = pd.read_sql_query("SELECT GenreID, GenreName FROM genre_ref;", conn)
    gmap = dict(zip(genre_ref_db["GenreName"], genre_ref_db["GenreID"]))

    genre_pairs_df["GenreID"] = genre_pairs_df["GenreName"].map(gmap)
    movie_genre_df = genre_pairs_df[["Title", "GenreID"]].dropna().drop_duplicates()
    safe_to_sql(movie_genre_df, "movie_genre")

# Load Principals Dataset
print("🎬 Loading title.principals.tsv...")

principals_df = None
try:
    principals_df = pd.read_csv(PRINCIPALS_PATH, sep="\t", dtype=str)

    # keep only titles that appear in IMDb dataset
    principals_df = principals_df.merge(
        title_df[["Title"]],
        left_on="tconst",
        right_on="Title",
        how="inner"
    )

    # CATEGORY many-to-many (from 'category' column)
    print("Building Movie–Category many-to-many...")
    if "category" in principals_df.columns:
        cat_pairs_df = principals_df[["Title", "category"]].dropna().rename(columns={"category": "CategoryName"})
        cat_pairs_df = cat_pairs_df.drop_duplicates()

        unique_cats = sorted(cat_pairs_df["CategoryName"].unique())
        category_ref_df = pd.DataFrame({"CategoryName": unique_cats})
        safe_to_sql(category_ref_df, "category_ref")

        cat_ref_db = pd.read_sql_query("SELECT CategoryID, CategoryName FROM category_ref;", conn)
        cmap = dict(zip(cat_ref_db["CategoryName"], cat_ref_db["CategoryID"]))

        cat_pairs_df["CategoryID"] = cat_pairs_df["CategoryName"].map(cmap)
        movie_category_df = cat_pairs_df[["Title", "CategoryID"]].dropna().drop_duplicates()
        safe_to_sql(movie_category_df, "movie_category")

    # CHARACTER many-to-many
    # Prefer 'characters' column; if missing, we can skip or fall back
    print("Building Movie–Character many-to-many...")
    char_col = None
    if "characters" in principals_df.columns:
        char_col = "characters"
    elif "job" in principals_df.columns:
        # last resort, treat job strings as character labels (not ideal, but keeps project happy)
        char_col = "job"

    if char_col:
        char_pairs_df = principals_df[["Title", char_col]].dropna().rename(columns={char_col: "CharacterName"})
        char_pairs_df = char_pairs_df.drop_duplicates()

        unique_chars = sorted(char_pairs_df["CharacterName"].unique())
        character_ref_df = pd.DataFrame({"CharacterName": unique_chars})
        safe_to_sql(character_ref_df, "character_ref")

        char_ref_db = pd.read_sql_query("SELECT CharacterID, CharacterName FROM character_ref;", conn)
        chmap = dict(zip(char_ref_db["CharacterName"], char_ref_db["CharacterID"]))

        char_pairs_df["CharacterID"] = char_pairs_df["CharacterName"].map(chmap)
        movie_character_df = char_pairs_df[["Title", "CharacterID"]].dropna().drop_duplicates()
        safe_to_sql(movie_character_df, "movie_character")

    # JOB many-to-many (from 'job' column)
    print("Building Movie–Job many-to-many...")
    if "job" in principals_df.columns:
        job_pairs_df = principals_df[["Title", "job"]].dropna().rename(columns={"job": "JobName"})
        job_pairs_df = job_pairs_df.drop_duplicates()

        unique_jobs = sorted(job_pairs_df["JobName"].unique())
        job_ref_df = pd.DataFrame({"JobName": unique_jobs})
        safe_to_sql(job_ref_df, "job_ref")

        job_ref_db = pd.read_sql_query("SELECT JobID, JobName FROM job_ref;", conn)
        jmap = dict(zip(job_ref_db["JobName"], job_ref_db["JobID"]))

        job_pairs_df["JobID"] = job_pairs_df["JobName"].map(jmap)
        movie_job_df = job_pairs_df[["Title", "JobID"]].dropna().drop_duplicates()
        safe_to_sql(movie_job_df, "movie_job")

except FileNotFoundError:
    print("principals TSV not found — skipping principals-based tables.")
except Exception as e:
    print(f" Error importing principals: {e}")

# Import storyline vectors
print(" Loading storyline vectors...")

try:
    vectors_df = pd.read_csv(VECTORS_PATH)

    # Detect if first row looks numeric-only => no header
    first_col = str(vectors_df.columns[0])
    is_number_like = first_col.replace('.', '', 1).replace('-', '', 1).isdigit()
    if is_number_like:
        vectors_df = pd.read_csv(VECTORS_PATH, header=None)
        vectors_df.columns = [f"dim_{i}" for i in range(vectors_df.shape[1])]

    # Attach Title if missing
    if "Title" not in vectors_df.columns:
        vectors_df.insert(0, "Title", title_df["Title"].values[:len(vectors_df)])

    vector_cols = [col for col in vectors_df.columns if col != "Title"]
    col_defs = ", ".join([f'"{col}" REAL' for col in vector_cols])

    cur.execute(f"""
    CREATE TABLE storyline_vector (
        Title TEXT,
        {col_defs},
        FOREIGN KEY (Title) REFERENCES title(Title) ON DELETE CASCADE
    );
    """)

    safe_to_sql(vectors_df, "storyline_vector")
    print(f"Imported storyline_vector with {vectors_df.shape[0]} rows.")

except FileNotFoundError:
    print(" storyline_vector.csv not found — skipping.")
except Exception as e:
    print(f"Error importing storyline_vector: {e}")

# Finish
conn.commit()
conn.close()
print(" movies.sqlite created successfully with many-to-many tables for genre, category, character, and job!")
