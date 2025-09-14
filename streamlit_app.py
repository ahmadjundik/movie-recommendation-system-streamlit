# app.py
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------
# Load dataset
# ---------------------
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")
links = pd.read_csv("dataset/links.csv")

# Hitung rating rata-rata & jumlah rating
ratings_summary = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

movies = pd.merge(movies, ratings_summary, on='movieId', how='left')
movies[['avg_rating', 'num_ratings']] = movies[['avg_rating', 'num_ratings']].fillna(0)

# Gabung dengan links.csv agar ada tmdbId
movies = pd.merge(movies, links[['movieId', 'tmdbId']], on='movieId', how='left')

# ---------------------
# Content-based filtering (TF-IDF + cosine similarity)
# ---------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(""))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Buat mapping index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    """Rekomendasi berdasarkan genre (content-based TF-IDF + cosine similarity)"""
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # ambil 5 film teratas
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'avg_rating', 'num_ratings', 'tmdbId']]

def get_top_movies_by_genre(genre, top_n=10):
    """10 film terbaik berdasarkan genre"""
    filtered = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    top_movies = filtered.sort_values(by=['avg_rating', 'num_ratings'], ascending=[False, False]).head(top_n)
    return top_movies[['title', 'avg_rating', 'num_ratings', 'tmdbId']]

# ---------------------
# Fungsi ambil poster dari TMDb
# ---------------------
TMDB_API_KEY = "6740125ed80fde295270a1f93cf0105a"  # ganti dengan TMDb API key kamu
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US"
TMDB_IMG_URL = "https://image.tmdb.org/t/p/w200"

def get_poster_url(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    url = TMDB_BASE_URL.format(int(tmdb_id), TMDB_API_KEY)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("poster_path"):
                return TMDB_IMG_URL + data["poster_path"]
    except:
        return None
    return None

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Rekomendasi film berdasarkan **MovieLens Dataset (Genres + Ratings)** dengan Content-Based Filtering")

# Pilih mode
mode = st.radio("Pilih jenis rekomendasi:", ["🔍 Berdasarkan Film", "⭐ Top 10 Film per Genre"])

if mode == "🔍 Berdasarkan Film":
    # Dropdown untuk memilih film
    movie_list = sorted(movies['title'].dropna().unique())
    selected_movie = st.selectbox("Pilih film:", movie_list)

    # Tombol untuk rekomendasi
    if st.button("Dapatkan Rekomendasi"):
        recommendations = get_recommendations(selected_movie)
        if not recommendations.empty:
            st.subheader("🎯 Rekomendasi Film:")
            cols = st.columns(5)
            for i, row in recommendations.reset_index(drop=True).iterrows():
                with cols[i % 5]:
                    poster_url = get_poster_url(row['tmdbId'])
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"⭐ {row['avg_rating']:.2f} ({int(row['num_ratings'])} ratings)")
        else:
            st.warning("Maaf, tidak ditemukan rekomendasi untuk film ini.")

elif mode == "⭐ Top 10 Film per Genre":
    # Dropdown genre unik
    all_genres = sorted(set([g for sublist in movies['genres'].dropna().str.split('|') for g in sublist]))
    selected_genre = st.selectbox("Pilih genre:", all_genres)

    # Tombol untuk tampilkan top 10
    if st.button("Tampilkan Top 10"):
        top_movies = get_top_movies_by_genre(selected_genre, top_n=10)
        if not top_movies.empty:
            st.subheader(f"🏆 Top 10 Film Terbaik dalam Genre **{selected_genre}**")
            cols = st.columns(5)
            for i, row in top_movies.reset_index(drop=True).iterrows():
                with cols[i % 5]:
                    poster_url = get_poster_url(row['tmdbId'])
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"⭐ {row['avg_rating']:.2f} ({int(row['num_ratings'])} ratings)")
        else:
            st.warning("Tidak ada film dengan genre ini.")

