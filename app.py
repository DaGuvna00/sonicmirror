import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheHandler

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

# Setup token cache
if "token_info" not in st.session_state:
    st.session_state.token_info = None

class StreamlitTokenCache(CacheHandler):
    def get_cached_token(self):
        return st.session_state.token_info

    def save_token_to_cache(self, token_info):
        st.session_state.token_info = token_info

# Auth manager
auth_manager = SpotifyOAuth(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="user-library-read playlist-read-private",
    cache_handler=StreamlitTokenCache()
)

# ✅ If redirected from Spotify with ?code=, process it manually
query_params = st.query_params
if "code" in query_params and st.session_state.token_info is None:
    code = query_params["code"]
    token_info = auth_manager.get_access_token(code, as_dict=True)
    st.session_state.token_info = token_info
    st.success("🎉 Spotify login completed! You can now use the app.")
    st.rerun()

# 🔐 If not logged in yet
if st.session_state.token_info is None:
    auth_url = auth_manager.get_authorize_url()
    st.warning("🔐 Please log in with Spotify to continue:")
    st.markdown(f"[Click here to log in with Spotify]({auth_url})")

else:
    # 🎧 Ready to go
    sp = spotipy.Spotify(auth=st.session_state.token_info['access_token'])
    user = sp.current_user()
    st.success(f"✅ Logged in as: {user['display_name']}")

    # Show playlists
    st.subheader("🎵 Your Spotify Playlists")
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        name = playlist['name']
        total = playlist['tracks']['total']
        st.markdown(f"- **{name}** ({total} tracks)")
