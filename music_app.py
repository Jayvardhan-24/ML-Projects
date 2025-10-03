import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------- SPOTIFY SETUP -----------------
CLIENT_ID = "YOUR_CLIENTID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"

client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID, 
    client_secret=CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# ----------------- FUNCTION TO FETCH ALBUM COVER -----------------
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

# ----------------- DATA PREPARATION -----------------
def prepare_data():
    """Load CSV, create features, compute similarity, and save pickle files."""
    music = pd.read_csv("spotify_millsongdata.csv", on_bad_lines="skip")
    music = music[['song', 'artist']]  # keep only needed columns
    music['text'] = music['song'] + " " + music['artist']  # simple feature for demo

    # Compute similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(music['text'])
    similarity = cosine_similarity(tfidf_matrix)

    # Save pickle files
    pickle.dump(music, open("df.pkl", "wb"))
    pickle.dump(similarity, open("similarity.pkl", "wb"))

    return music, similarity

# ----------------- LOAD DATA -----------------
if os.path.exists("df.pkl") and os.path.exists("similarity.pkl"):
    music = pickle.load(open("df.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
else:
    music, similarity = prepare_data()

# ----------------- RECOMMENDATION FUNCTION -----------------
def recommend(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []

    with st.spinner("Fetching recommendations and album covers..."):
        for i in distances[1:6]:
            artist = music.iloc[i[0]].artist
            recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
            recommended_music_names.append(music.iloc[i[0]].song)
    return recommended_music_names, recommended_music_posters

# ----------------- STREAMLIT APP -----------------
st.header('Music Recommender System')

music_list = music['song'].values
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters = recommend(selected_song)
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        col.text(recommended_music_names[idx])
        col.image(recommended_music_posters[idx])
