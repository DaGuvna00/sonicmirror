import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

st.write("🟢 App loaded successfully.")

try:
    st.write("🔐 Setting up Spotify auth...")
    auth_manager = SpotifyOAuth(
        client_id=st.secrets["SPOTIPY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
        scope="user-library-read playlist-read-private"
    )

    st.write("🔄 Attempting to connect to Spotify...")
    sp = spotipy.Spotify(auth_manager=auth_manager)

    st.write("👤 Getting current user...")
    user = sp.current_user()
    st.success(f"✅ Logged in as: {user['display_name']}")

    # Show playlists
    st.subheader("🎵 Your Spotify Playlists")
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        st.markdown(f"- **{playlist['name']}** ({playlist['tracks']['total']} tracks)")

except Exception as e:
    st.warning("⚠️ Spotify authentication failed or login not completed.")
    st.markdown("Please try logging in below:")
    try:
        auth_url = auth_manager.get_authorize_url()
        st.markdown(f"[Click here to log in with Spotify]({auth_url})")
    except Exception as inner:
        st.error(f"Critical Error: {inner}")

    st.error(f"Error: {e}")
