import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Load Dataset
# =============================
@st.cache_data
def load_data():
    movies = pd.read_csv("dataset/movies.csv")
    ratings = pd.read_csv("dataset/ratings.csv")
    tags = pd.read_csv("dataset/tags.csv")
    return movies, ratings, tags

movies, ratings, tags = load_data()

# =============================
# Content-Based Recommendation
# =============================
@st.cache_data
def build_similarity_matrix():
    # Gabungkan judul + genres
    movies['combined'] = movies['title'] + " " + movies['genres']
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_similarity_matrix()

# =============================
# Fungsi rekomendasi
# =============================
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'].str.contains(title, case=False, na=False)].index
    if len(idx) == 0:
        return None
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # 10 rekomendasi

    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

# =============================
# Streamlit UI
# =============================
st.title("🎬 Movie Recommendation System")
st.write("Berbasis **MovieLens Dataset** (Content-Based Filtering)")

# Input judul film
movie_name = st.text_input("Masukkan judul film:")
if movie_name:
    recommendations = get_recommendations(movie_name)
    if recommendations is not None:
        st.subheader("Rekomendasi film mirip:")
        st.table(recommendations)
    else:
        st.warning("Film tidak ditemukan di dataset!")
