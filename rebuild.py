import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

DB_PATH = "movies.sqlite"

# 1. Load the correct GTE model
def load_model():
    model_name = "Alibaba-NLP/gte-base-en-v1.5"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model: {model_name} on {device.upper()}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return model

# 2. Load storylines from DB
def load_storylines():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT Title, Synopsis
        FROM synopsis
        ORDER BY Title;
    """, conn)

    conn.close()

    print(f"Loaded {len(df)} storylines.")
    return df

# 3. Generate new 768-dim embeddings
def embed_all(df, model):
    texts = df["Synopsis"].fillna("").tolist()
    print("Encoding storylines...")

    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    print(f"Generated embeddings shape: {vecs.shape}")

    return vecs.astype(np.float32)

# 4. Replace the storyline_vector table with NEW embeddings
def save_to_db(df, vectors):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("Dropping old storyline_vector...")
    cur.execute("DROP TABLE IF EXISTS storyline_vector;")

    # Create new table with 768 embedding columns
    cols = ", ".join([f"dim_{i} REAL" for i in range(768)])

    create_sql = f"""
        CREATE TABLE storyline_vector (
            Title TEXT PRIMARY KEY,
            {cols}
        );
    """

    cur.execute(create_sql)
    print("Created new storyline_vector table.")

    # Insert embeddings
    print("Inserting new embeddings...")
    for i, title in enumerate(df["Title"]):
        values = [title] + [float(x) for x in vectors[i]]
        placeholders = ", ".join(["?"] * (1 + 768))
        cur.execute(f"INSERT INTO storyline_vector VALUES ({placeholders})", values)

        if i % 200 == 0:
            print(f"Inserted {i} rows...")

    conn.commit()
    conn.close()
    print("All embeddings saved successfully!")

# MAIN PIPELINE
if __name__ == "__main__":
    model = load_model()
    df = load_storylines()
    vectors = embed_all(df, model)
    save_to_db(df, vectors)
    print("🎉 Embedding rebuild complete!")
