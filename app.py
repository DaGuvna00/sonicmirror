import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎵 SonicMirror – Spotify Playlist Analyzer")

# ---------------------------
# Spotify Auth Setup
# ---------------------------
auth_manager = SpotifyOAuth(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="playlist-read-private playlist-read-collaborative"
)

sp = spotipy.Spotify(auth_manager=auth_manager)

# ---------------------------
# Show login confirmation
# ---------------------------
user = sp.current_user()
st.success(f"✅ Logged in as: {user['display_name']}")

# ---------------------------
# Fetch and show playlists
# ---------------------------
st.subheader("🧩 Select Playlists to Analyze")

playlists = sp.current_user_playlists()["items"]
playlist_names = [p["name"] for p in playlists]
playlist_map = {p["name"]: p["id"] for p in playlists}

selected_names = st.multiselect("Choose one or more playlists", playlist_names)

# ---------------------------
# If selected, Analyze
# ---------------------------
if selected_names and st.button("🔍 Analyze Selected Playlists"):
    st.write("🔁 Analyze button clicked")
    all_tracks = []
    st.write("🧠 Tracks found:", len(all_tracks))

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
                if track and track["id"]:  # Skip empty/null tracks
                    all_tracks.append({
                        "playlist": name,
                        "name": track["name"],
                        "id": track["id"],
                        "artists": ", ".join([a["name"] for a in track["artists"]]),
                        "album": track["album"]["name"]
                    })
            offset += 100

    df = pd.DataFrame(all_tracks)
    st.write("✅ Tracks collected:", len(df))
    st.write("🎧 Playlist Sample:")
    st.dataframe(df.head())


