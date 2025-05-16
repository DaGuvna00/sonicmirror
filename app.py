import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

# Authenticate with Spotify using secrets
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="user-library-read playlist-read-private"
))

# Fetch current user's profile
user = sp.current_user()
st.success(f"Logged in as: {user['display_name']}")

# Fetch playlists
st.subheader("🎵 Your Spotify Playlists")
playlists = sp.current_user_playlists()
for playlist in playlists['items']:
    name = playlist['name']
    total = playlist['tracks']['total']
    st.markdown(f"- **{name}** ({total} tracks)")
