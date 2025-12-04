AI Movie Search (AIMovieSearch)

AI Movie Search is a Streamlit web app that uses FAISS and TF-IDF to find movies with similar storylines.
It also includes a spell checker and shows movie posters from TMDB.

How to Run:
Install the required libraries:
pip install streamlit faiss-cpu numpy pandas scikit-learn requests joblib pyspellchecker

Then, you will need to download the db separately.

Run the app:
streamlit run main.py
Open the link shown in the terminal (usually http://localhost:8501).

Notes:
The first time you run the app, it might take a few minutes to load data and build the FAISS index.
After that, searches will run much faster.

Features:
- Search for movies by plot or idea
- Finds similar storylines using FAISS
- Built-in spell correction
- Shows movie posters from TMDB
- Caches data for faster performance

Developers
for this CSE 111 Project:
Richard Camacho,
Akshaya Natarajan,
Ailisha Shukla
