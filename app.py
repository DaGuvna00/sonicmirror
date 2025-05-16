import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheHandler
import pandas as pd

# ---- Page Setup ----
st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

# ---- Token + Query Handling ----
if "token_info" not in st.session_state:
    st.session_state.token_info = None

if "code" in st.query_params and st.session_state.token_info is None:
    code = st.query_params["code"]
    auth_manager = SpotifyOAuth(
        client_id=st.secrets["SPOTIPY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
        scope="user-library-read playlist-read-private"
    )
    token_info = auth_manager.get_access_token(code, as_dict=True)
    st.session_state.token_info = token_info
    st.rerun()

# ---- Login Flow ----
if st.session_state.token_info is None:
    auth_manager = SpotifyOAuth(
        client_id=st.secrets["SPOTIPY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
        scope="user-library-read playlist-read-private"
    )
    auth_url = auth_manager.get_authorize_url()
    st.warning("🔐 Please log in with Spotify to continue:")
    st.markdown(f"[Click here to log in with Spotify]({auth_url})")

# ---- If Logged In ----
else:
    sp = spotipy.Spotify(auth=st.session_state.token_info['access_token'])
    user = sp.current_user()
    st.success(f"✅ Logged in as: {user['display_name']}")

    # --- Playlist Selection ---
    playlists = sp.current_user_playlists()['items']
    playlist_names = [p['name'] for p in playlists]
    playlist_map = {p['name']: p['id'] for p in playlists}

    st.subheader("🎛 Select Playlists to Analyze")
    selected_names = st.multiselect("Choose one or more playlists", playlist_names)

    # --- If selected and button clicked ---
    if selected_names and st.button("Analyze Selected Playlists"):
        st.write("🎯 Analyze button clicked")
        all_tracks = []
        st.write("📦 Tracks found:", len(all_tracks))

    for name in selected_names:
    playlist_id = playlist_map[name]
    offset = 0
    while True:
        results = sp.playlist_tracks(playlist_id, offset=offset, limit=100)
        tracks = results['items']
        if not tracks:
            break

    for item in tracks:
            track = item['track']
            if track and track["id"]:
                all_tracks.append({
                    "playlist": name,
                    "name": track["name"],
                    "id": track["id"],
                    "artists": ", ".join([a["name"] for a in track["artists"]]),
                    "album": track["album"]["name"]
                })

        offset += 100


        # ---- Batching for Audio Features (max 100 per request) ----
def get_audio_features_in_batches(track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        try:
            features = sp.audio_features(batch)
            audio_features.extend(features)
        except Exception as e:
            st.error(f"⚠️ Error fetching batch {i//100+1}: {e}")
    return audio_features

    track_ids = [t["id"] for t in all_tracks if t["id"]]  # Filter out missing/null track IDs

    audio_features = get_audio_features_in_batches(track_ids)
    st.write("✅ Got audio features for", len(track_ids), "tracks.")

    for i, features in enumerate(audio_features):
        if features:
            all_tracks[i].update(features)

    df = pd.DataFrame(all_tracks)
    st.write("📊 Final DataFrame rows:", df.shape[0])

    st.write("📊 DataFrame shape:", df.shape)

    st.dataframe(df[["playlist", "name", "artists", "energy", "valence", "danceability", "tempo"]])
    st.session_state.df = df


