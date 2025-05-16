import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

# ✅ DEBUG CHECKPOINT
st.write("✅ App is running!")

# Set up Spotify OAuth
try:
    auth_manager = SpotifyOAuth(
        client_id=st.secrets["SPOTIPY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
        scope="user-library-read playlist-read-private"
    )

    # ✅ DEBUG CHECKPOINT
    st.write("🔐 Auth manager created")

    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()

    # ✅ DEBUG CHECKPOINT
    st.success(f"✅ Logged in as: {user['display_name']}")

    # Fetch and show playlists
    st.subheader("🎵 Your Spotify Playlists")
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        name = playlist['name']
        total = playlist['tracks']['total']
        st.markdown(f"- **{name}** ({total} tracks)")

except Exception as e:
    st.warning("⚠️ Could not log in or load playlists.")
    st.error(f"Error: {e}")
    auth_url = auth_manager.get_authorize_url()
    st.markdown(f"[Click here to log in with Spotify]({auth_url})")
