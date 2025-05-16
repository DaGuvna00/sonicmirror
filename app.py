import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheHandler

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

# ✅ Trigger rerun after Spotify redirects with ?code=...
if st.experimental_get_query_params().get("code") and "reran" not in st.session_state:
    st.session_state.reran = True
    st.experimental_rerun()

# Setup in-memory token cache
if "token_info" not in st.session_state:
    st.session_state.token_info = None

class StreamlitTokenCache(CacheHandler):
    def get_cached_token(self):
        return st.session_state.token_info

    def save_token_to_cache(self, token_info):
        st.session_state.token_info = token_info

# Setup Spotify auth
auth_manager = SpotifyOAuth(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="user-library-read playlist-read-private",
    cache_handler=StreamlitTokenCache()
)

# Run login
if not auth_manager.get_cached_token():
    auth_url = auth_manager.get_authorize_url()
    st.warning("🔐 Please log in with Spotify to continue:")
    st.markdown(f"[Click here to log in with Spotify]({auth_url})")
else:
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
    st.success(f"✅ Logged in as: {user['display_name']}")

    # Show playlists
    st.subheader("🎵 Your Spotify Playlists")
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        st.markdown(f"- **{playlist['name']}** ({playlist['tracks']['total']} tracks)")

