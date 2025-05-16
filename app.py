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

    # 🎯 Get all playlists
playlists = sp.current_user_playlists()['items']
playlist_names = [p['name'] for p in playlists]
playlist_map = {p['name']: p['id'] for p in playlists}

# ✅ Let the user select multiple playlists
st.subheader("🎛 Select Playlists to Analyze")
selected_names = st.multiselect("Choose one or more playlists", playlist_names)

if selected_names:
    if st.button("Analyze Selected Playlists"):
        all_tracks = []

        # 🔁 Go through each selected playlist
        for name in selected_names:
            playlist_id = playlist_map[name]
            results = sp.playlist_tracks(playlist_id)
            tracks = results['items']

            # 🔽 Flatten track info
            for item in tracks:
                track = item['track']
                if track:  # safety check
                    all_tracks.append({
                        "playlist": name,
                        "name": track["name"],
                        "id": track["id"],
                        "artists": ", ".join([a["name"] for a in track["artists"]]),
                        "album": track["album"]["name"]
                    })

        # 🎵 Get audio features
        track_ids = [t["id"] for t in all_tracks if t["id"]]
        audio_features = sp.audio_features(track_ids)

        # 🧠 Merge features with track metadata
        for i, features in enumerate(audio_features):
            if features:
                all_tracks[i].update(features)

        import pandas as pd
        df = pd.DataFrame(all_tracks)
        st.success(f"✅ Loaded {len(df)} tracks across {len(selected_names)} playlist(s).")

        # 👀 Preview
        st.dataframe(df[["playlist", "name", "artists", "energy", "valence", "danceability", "tempo"]])

        # 🎯 Store it in session state for use in other pages or charts
        st.session_state.df = df

    # Show playlists
    st.subheader("🎵 Your Spotify Playlists")
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        name = playlist['name']
        total = playlist['tracks']['total']
        st.markdown(f"- **{name}** ({total} tracks)")
