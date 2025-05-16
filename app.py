import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

# --- Setup Auth Manager ---
auth_manager = SpotifyOAuth(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="playlist-read-private playlist-read-collaborative user-library-read"
)

# --- Connect to Spotify ---
sp = None
try:
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
    st.success(f"✅ Logged in as: {user['display_name']}")
except Exception as e:
    st.warning("🔐 Not logged in or token expired.")
    try:
        auth_url = auth_manager.get_authorize_url()
        st.markdown(f"[Click here to log in with Spotify]({auth_url})")
    except Exception as inner:
        st.error(f"Error creating login link: {inner}")
    st.stop()


# --- Playlist Selection ---
playlists = sp.current_user_playlists()["items"]
playlist_names = [p["name"] for p in playlists]
playlist_map = {p["name"]: p["id"] for p in playlists}

st.subheader("🎛 Select Playlists to Analyze")
selected_names = st.multiselect("Choose one or more playlists", playlist_names)

# --- If selected and button clicked ---
if selected_names and st.button("Analyze Selected Playlists"):
    all_tracks = []

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

    # --- Audio Features ---
    def get_audio_features_in_batches(track_ids):
        audio_features = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            audio_features.extend(sp.audio_features(batch))
        return audio_features

    track_ids = [t["id"] for t in all_tracks if t["id"]]
    audio_features = get_audio_features_in_batches(track_ids)

    for i, features in enumerate(audio_features):
        if features:
            all_tracks[i].update(features)

    df = pd.DataFrame(all_tracks)
    st.success(f"🎉 Loaded {len(df)} tracks from {len(selected_names)} playlist(s).")
    st.dataframe(df[["playlist", "name", "artists", "energy", "valence", "danceability", "tempo"]])
    st.session_state.df = df
