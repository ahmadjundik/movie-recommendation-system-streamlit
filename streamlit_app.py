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
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # ambil 5 film teratas
    movie_indices = [i[0] for i in sim_scores]
    return movies_tags.iloc[movie_indices][['title', 'avg_rating', 'num_ratings']]

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Movie Recommendation System", layout="centered", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Rekomendasi film berdasarkan **MovieLens Dataset (Tags + Ratings)**")

# Dropdown untuk memilih film
movie_list = movies_tags['title'].values
selected_movie = st.selectbox("Pilih film:", movie_list)

# Tombol untuk rekomendasi
if st.button("Dapatkan Rekomendasi"):
    recommendations = get_recommendations(selected_movie)
    if not recommendations.empty:
        st.subheader("🎯 Rekomendasi Film:")
        for i, row in recommendations.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"⭐ Rata-rata rating: {row['avg_rating']:.2f}")
            st.write(f"👥 Jumlah rating: {int(row['num_ratings'])}")
            st.markdown("---")
    else:
        st.warning("Maaf, tidak ditemukan rekomendasi untuk film ini.")
