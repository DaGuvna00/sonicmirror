import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Page config ---
st.set_page_config(page_title="SonicMirror - Playlist Analyzer", layout="wide")

# --- Spotify Setup ---
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

# --- Helper ---
def chunked(iterable, size=100):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# --- UI ---
st.title("🎶 SonicMirror – Analyze Your Spotify Playlists")
auth_url = sp_oauth.get_authorize_url()
st.markdown(f"[🔐 Log in with Spotify]({auth_url})")

all_data = []

code = st.query_params.get("code")
if code:
    token_info = sp_oauth.get_access_token(code)
    if token_info:
        access_token = token_info["access_token"]
        sp = spotipy.Spotify(auth=access_token)

        user = sp.current_user()
        st.success(f"Logged in as {user['display_name']}")

        playlists = sp.current_user_playlists(limit=50)
        names = [p["name"] for p in playlists["items"]]
        ids = [p["id"] for p in playlists["items"]]

        sel = st.selectbox("🎧 Choose a Playlist", names)
        if sel:
            idx = names.index(sel)
            playlist_id = ids[idx]
            data = sp.playlist_tracks(playlist_id)

            track_ids, names, artists = [], [], []
            for item in data["items"]:
                track = item["track"]
                if track and track["id"]:
                    track_ids.append(track["id"])
                    names.append(track["name"])
                    artists.append(", ".join([a["name"] for a in track["artists"]]))

            features = []
            for chunk in chunked(track_ids):
                try:
                    feats = sp.audio_features(chunk)
                    if feats:
                        features.extend([f for f in feats if f])
                except Exception as e:
                    st.warning(f"⚠️ Skipped a chunk due to error: {e}")

            if features:
                df = pd.DataFrame(features)
                df["Track Name"] = names[:len(df)]
                df["Artist Name(s)"] = artists[:len(df)]
                df["Playlist"] = sel
                all_data.append(df)
            else:
                st.warning("⚠️ No usable features returned from Spotify for this playlist.")

# --- Upload from Exportify ---
uploaded_files = st.file_uploader("📂 Upload Exportify files (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith("csv"):
                data = pd.read_csv(file)
            else:
                data = pd.read_excel(file)
            data["Playlist"] = file.name
            all_data.append(data)
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {e}")

# --- Charts ---
if all_data:
    df = pd.concat(all_data, ignore_index=True)
    df = df.dropna(subset=["Track Name", "Artist Name(s)"])
    st.subheader("📋 Playlist Overview")
    st.dataframe(df[["Track Name", "Artist Name(s)", "Playlist"]].head())

    metrics = ["energy", "valence", "danceability", "acousticness", "instrumentalness", "liveness", "tempo"]
    available = [m for m in metrics if m in df.columns]

    if available:
        st.subheader("📊 Feature Averages")
        st.dataframe(df[available].mean().round(3).rename("Average").to_frame())

        st.subheader("🧭 Radar Profile")
        radar_vals = df[available].mean().tolist()
        angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist() + [0]
        radar_vals += radar_vals[:1]
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        ax.plot(angles, radar_vals, "o-", linewidth=2)
        ax.fill(angles, radar_vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available)
        ax.set_title("Average Audio Profile")
        st.pyplot(fig)

        if "tempo" in df.columns:
            st.subheader("🎵 Tempo Histogram")
            fig2, ax2 = plt.subplots()
            ax2.hist(df["tempo"].dropna(), bins=30)
            st.pyplot(fig2)

    if "Artist Name(s)" in df.columns:
        st.subheader("☁️ Artist Word Cloud")
        text = " ".join(df["Artist Name(s)"].dropna().astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wc.to_array(), use_container_width=True)
