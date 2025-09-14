# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------
# Load dataset
# ---------------------
movies = pd.read_csv("dataset/movies.csv")
tags = pd.read_csv("dataset/tags.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Gabungkan movies dengan tags
tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
movies_tags = pd.merge(movies, tags_grouped, on='movieId', how='left')
movies_tags['tag'] = movies_tags['tag'].fillna("")

# Hitung rating rata-rata & jumlah rating
ratings_summary = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

movies_tags = pd.merge(movies_tags, ratings_summary, on='movieId', how='left')
movies_tags[['avg_rating', 'num_ratings']] = movies_tags[['avg_rating', 'num_ratings']].fillna(0)

# ---------------------
# Content-based filtering (TF-IDF + cosine similarity)
# ---------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_tags['tag'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Buat mapping index
indices = pd.Series(movies_tags.index, index=movies_tags['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    """Rekomendasi berdasarkan film"""
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # ambil 5 film teratas
    movie_indices = [i[0] for i in sim_scores]
    return movies_tags.iloc[movie_indices][['title', 'avg_rating', 'num_ratings']]

def get_top_movies_by_genre(genre, top_n=10):
    """10 film terbaik berdasarkan genre (dengan avg_rating & num_ratings)"""
    filtered = movies_tags[movies_tags['genres'].str.contains(genre, case=False, na=False)]
    top_movies = filtered.sort_values(by=['avg_rating', 'num_ratings'], ascending=[False, False]).head(top_n)
    return top_movies[['title', 'avg_rating', 'num_ratings']]

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Movie Recommendation System", layout="centered", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Rekomendasi film berdasarkan **MovieLens Dataset (Tags + Ratings)**")

# Pilih mode
mode = st.radio("Pilih jenis rekomendasi:", ["🔍 Berdasarkan Film", "⭐ Top 10 Film per Genre"])

if mode == "🔍 Berdasarkan Film":
    # Dropdown untuk memilih film
    movie_list = movies_tags['title'].values
    selected_movie = st.selectbox("Pilih film:", movie_list)

    # Tombol untuk rekomendasi
    if st.button("Dapatkan Rekomendasi"):
        recommendations = get_recommendations(selected_movie)
        if not recommendations.empty:
            st.subheader("🎯 Rekomendasi Film:")
            st.dataframe(recommendations.reset_index(drop=True))
        else:
            st.warning("Maaf, tidak ditemukan rekomendasi untuk film ini.")

elif mode == "⭐ Top 10 Film per Genre":
    # Dropdown genre unik
    all_genres = sorted(set([g for sublist in movies['genres'].str.split('|') for g in sublist]))
    selected_genre = st.selectbox("Pilih genre:", all_genres)

    # Tombol untuk tampilkan top 10
    if st.button("Tampilkan Top 10"):
        top_movies = get_top_movies_by_genre(selected_genre, top_n=10)
        if not top_movies.empty:
            st.subheader(f"🏆 Top 10 Film Terbaik dalam Genre **{selected_genre}**")
            st.dataframe(top_movies.reset_index(drop=True))
        else:
            st.warning("Tidak ada film dengan genre ini.")
