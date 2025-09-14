import requests

API_KEY = "YOUR_TMDB_API_KEY"  # ganti dengan API key TMDb kamu
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def get_movie_poster(movie_id, links_df):
    """
    Ambil poster film dari TMDb berdasarkan movieId.
    links.csv harus punya kolom: movieId, tmdbId
    """
    try:
        tmdb_id = links_df.loc[links_df['movieId'] == movie_id, 'tmdbId'].values[0]
        url = f"{BASE_URL}/movie/{tmdb_id}?api_key={API_KEY}"
        response = requests.get(url).json()
        poster_path = response.get("poster_path")
        if poster_path:
            return IMAGE_BASE_URL + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"
