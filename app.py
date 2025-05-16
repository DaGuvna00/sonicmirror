import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")
st.write("👀 Reached this point in the app...")

# Set up Spotify OAuth
auth_manager = SpotifyOAuth(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="user-library-read playlist-read-private"
)

try:
    # Try to authenticate and fetch user
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
   st.success(f"✅ Logged in as: {user['display_name']}")


    # Fetch and show playlists
    st.subheader("🎵 Your Spotify Playlists")
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        name = playlist['name']
        total = playlist['tracks']['total']
        st.markdown(f"- **{name}** ({total} tracks)")

except:
    st.warning("🔐 Please log in to Spotify using the link below:")
    auth_url = auth_manager.get_authorize_url()
    st.markdown(f"[Click here to log in with Spotify]({auth_url})")
