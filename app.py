
import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="SonicMirror - Playlist Analyzer", layout="wide")

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Spotify Auth Setup ---
SPOTIPY_CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]
SPOTIPY_REDIRECT_URI = "https://sonicmirror.streamlit.app"

scope = "playlist-read-private playlist-read-collaborative"

sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=scope,
    show_dialog=True
)

# --- Helper Function ---
def chunked(iterable, size=80):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# --- UI ---
st.title("🎶 SonicMirror – Analyze Your Spotify Playlists")
auth_url = sp_oauth.get_authorize_url()
st.markdown(f"[🔐 Log in with Spotify]({auth_url})")

# --- Spotify Logic ---
code = st.query_params.get("code")
all_data = []

if code:
    token_info = sp_oauth.get_access_token(code)
    if token_info:
        access_token = token_info["access_token"]
        sp = spotipy.Spotify(auth=access_token)

        user = sp.current_user()
        st.success(f"Logged in as {user['display_name']}")

        playlists = sp.current_user_playlists(limit=50)
        playlist_names = [pl['name'] for pl in playlists['items']]
        playlist_ids = [pl['id'] for pl in playlists['items']]

        selected = st.multiselect("🎧 Choose One or More Playlists", playlist_names)

        for sel in selected:
            idx = playlist_names.index(sel)
            playlist_id = playlist_ids[idx]
            tracks_data = sp.playlist_tracks(playlist_id)

            track_ids, track_names, artists, albums = [], [], [], []

            for item in tracks_data["items"]:
                track = item["track"]
                if track and track.get("id"):
                    track_ids.append(track["id"])
                    track_names.append(track["name"])
                    artists.append(", ".join([a["name"] for a in track["artists"]]))
                    albums.append(track["album"]["name"])

            audio_features = []
            for chunk in chunked(track_ids):
                try:
                    features = sp.audio_features(chunk)
                    clean = [f for f in features if f]
                    audio_features.extend(clean)
                except spotipy.SpotifyException as e:
                    st.warning(f"⚠️ Skipped a chunk due to error: {e}")

            if audio_features:
                df = pd.DataFrame(audio_features)
                df["Track Name"] = track_names[:len(df)]
                df["Artist Name(s)"] = artists[:len(df)]
                df["Album Name"] = albums[:len(df)]
                df["Playlist"] = sel

                all_data.append(df)
            else:
                st.warning(f"⚠️ No usable features returned from Spotify for this playlist.")

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload Exportify files (CSV/XLSX)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    all_files = []
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            f = pd.read_csv(file)
        else:
            f = pd.read_excel(file)
        f["Playlist"] = file.name
        all_files.append(f)

    if all_files:
        df = pd.concat(all_files, ignore_index=True)
        df = df.dropna(subset=["Track Name", "Artist Name(s)"])
        df = df[df["Duration (ms)"] > 0]
        all_data.append(df)

# --- Chart Rendering ---
if all_data:
    df = pd.concat(all_data, ignore_index=True)

    st.subheader("📋 Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df[["Track Name", "Artist Name(s)", "Playlist"]].head())

    # Radar
    metrics = ["energy", "valence", "danceability", "acousticness", "instrumentalness", "liveness"]
    available = [m for m in metrics if m in df.columns]
    if available:
        grouped = df.groupby("Playlist")[available].mean()
        angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        for name, row in grouped.iterrows():
            values = row.tolist() + [row.tolist()[0]]
            ax.plot(angles, values, label=name)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available)
        ax.set_title("🎯 Audio Profile by Playlist")
        ax.legend()
        st.pyplot(fig)

    # Tempo
    if "tempo" in df.columns:
        st.subheader("🎵 Tempo Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["tempo"].dropna(), bins=30)
        ax2.set_xlabel("Tempo (BPM)")
        ax2.set_ylabel("Tracks")
        st.pyplot(fig2)
