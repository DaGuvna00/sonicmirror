import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

st.markdown("🔗 [Click here to log in with Spotify](https://accounts.spotify.com/authorize?client_id=" + st.secrets["SPOTIPY_CLIENT_ID"] + "&response_type=code&redirect_uri=" + st.secrets["SPOTIPY_REDIRECT_URI"] + "&scope=user-library-read%20playlist-read-private)")

# Now attempt auth
auth_manager = SpotifyOAuth(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="user-library-read playlist-read-private"
)

sp = spotipy.Spotify(auth_manager=auth_manager)

try:
    user = sp.current_user()
    st.success(f"✅ Logged in as: {user['display_name']}")
except Exception as e:
    st.warning("⚠️ Not logged in yet.")
    st.error(f"Details: {e}")
