from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Deep-Learning Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
MOVIE_FILE = BASE_DIR / "movie_dict.pkl"
EMBEDDINGS_FILE = BASE_DIR / "movie_embeddings.npy"


@st.cache_resource
def load_data():
    if not MOVIE_FILE.exists():
        raise FileNotFoundError(
            "movie_dict.pkl was not found beside app_deep_learning.py"
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "movie_embeddings.npy was not found beside app_deep_learning.py"
        )

    with open(MOVIE_FILE, "rb") as file:
        movie_data = pickle.load(file)

    movies = (
        movie_data.copy()
        if isinstance(movie_data, pd.DataFrame)
        else pd.DataFrame(movie_data)
    )
    movies.reset_index(drop=True, inplace=True)

    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    required_columns = {"movie_id", "title"}
    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"movie_dict.pkl is missing columns: {sorted(missing_columns)}"
        )

    if len(movies) != len(embeddings):
        raise ValueError(
            "movie_dict.pkl and movie_embeddings.npy have different row counts."
        )

    return movies, embeddings


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(movie_id):
    url = (
        "https://api.themoviedb.org/3/movie/{}"
        "?api_key=8265bd1679663a7ea12ac168da84d2e8"
        "&language=en-US"
    ).format(movie_id)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        if not poster_path:
            return None

        return "https://image.tmdb.org/t/p/w500/" + poster_path

    except (requests.RequestException, ValueError):
        return None


def recommend(movie_title, number_of_results=5):
    matches = movies.index[movies["title"] == movie_title].tolist()

    if not matches:
        return []

    movie_index = matches[0]

    # The notebook saves normalized embeddings.
    # Dot product therefore gives cosine similarity.
    scores = embeddings @ embeddings[movie_index]
    ranked_indexes = np.argsort(-scores)

    recommendations = []

    for index in ranked_indexes:
        if index == movie_index:
            continue

        row = movies.iloc[index]

        recommendations.append(
            {
                "title": row["title"],
                "movie_id": int(row["movie_id"]),
                "poster": fetch_poster(row["movie_id"]),
                "similarity": float(scores[index]),
            }
        )

        if len(recommendations) == number_of_results:
            break

    return recommendations


try:
    movies, embeddings = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


st.title("🎬 Deep-Learning Movie Recommender System")
st.write(
    "Select a movie to receive five semantic recommendations generated "
    "from Sentence Transformer embeddings."
)

movie_titles = movies["title"].dropna().astype(str).tolist()

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_titles,
)

if st.button("Recommend", type="primary"):
    with st.spinner("Finding semantically similar movies..."):
        results = recommend(selected_movie)

    if not results:
        st.warning("No recommendations were found.")
    else:
        columns = st.columns(5, gap="medium")

        for column, result in zip(columns, results):
            with column:
                st.markdown(f"#### {result['title']}")

                if result["poster"]:
                    st.image(result["poster"], width="stretch")
                else:
                    st.info("Poster unavailable")

                st.caption(
                    f"Similarity: {result['similarity']:.3f}"
                )


st.divider()
st.caption(
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
