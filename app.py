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

# --- UI ---
st.title("\U0001F3B6 SonicMirror – Analyze Your Spotify Playlists")

auth_url = sp_oauth.get_authorize_url()
st.markdown(f"[\U0001F511 Log in with Spotify]({auth_url})")

# Initialize DataFrame
df = None

# --- Spotify Logic ---
code = st.query_params.get("code")

if code:
    token_info = sp_oauth.get_access_token(code)
    if token_info:
        access_token = token_info['access_token']
        sp = spotipy.Spotify(auth=access_token)

        user = sp.current_user()
        st.success(f"Logged in as {user['display_name']}")

        playlists = sp.current_user_playlists(limit=50)
        playlist_names = [pl['name'] for pl in playlists['items']]
        playlist_ids = [pl['id'] for pl in playlists['items']]

        selected = st.selectbox("\U0001F3A7 Choose a Playlist", playlist_names)

        if selected:
            idx = playlist_names.index(selected)
            playlist_id = playlist_ids[idx]
            tracks_data = sp.playlist_tracks(playlist_id)

            track_ids = []
            track_names = []
            artists = []

            for item in tracks_data['items']:
                track = item['track']
                if track and track['id']:
                    track_ids.append(track['id'])
                    track_names.append(track['name'])
                    artists.append(", ".join([a['name'] for a in track['artists']]))

            audio_features = sp.audio_features(track_ids)

            df = pd.DataFrame(audio_features)
            df["Track Name"] = track_names
            df["Artist Name(s)"] = artists
            df["Playlist"] = selected

            df.rename(columns={
                "energy": "Energy",
                "valence": "Valence",
                "danceability": "Danceability",
                "acousticness": "Acousticness",
                "instrumentalness": "Instrumentalness",
                "speechiness": "Speechiness",
                "liveness": "Liveness",
                "tempo": "Tempo",
                "loudness": "Loudness",
                "key": "Key"
            }, inplace=True)

# --- Exportify Upload Logic ---
uploaded_files = st.file_uploader(
    "Upload one or more Exportify playlist files (CSV or Excel)",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    all_dfs = []

    for file in uploaded_files:
        filename = file.name.rsplit(".", 1)[0]
        if file.name.endswith(".csv"):
            temp_df = pd.read_csv(file)
        else:
            xls = pd.ExcelFile(file)
            temp_df = pd.concat([xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True)

        temp_df["Playlist"] = filename
        all_dfs.append(temp_df)

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(subset=["Track Name", "Artist Name(s)"])
    df = df[df["Duration (ms)"] > 0]

# --- Shared Chart Section ---
if df is not None:
    st.subheader("\U0001F4CB Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df.head())

    # Averages
    st.subheader("\U0001F3DB Key Audio Feature Averages")
    metrics = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness", "Tempo"]
    available_metrics = [m for m in metrics if m in df.columns]
    st.dataframe(df[available_metrics].mean().round(3).rename("Average").to_frame())

    # Mood Map
    if "Energy" in df.columns and "Valence" in df.columns:
        st.subheader("\U0001F3A8 Mood Map: Energy vs Valence")
        fig1, ax1 = plt.subplots()
        ax1.scatter(df["Energy"], df["Valence"], alpha=0.5)
        ax1.set_xlabel("Energy")
        ax1.set_ylabel("Valence")
        ax1.set_title("Track Mood Distribution")
        st.pyplot(fig1)

    # Radar Chart
    radar_labels = [m for m in ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness"] if m in df.columns]
    if radar_labels:
        st.subheader("\U0001F5B8 Audio Feature Profile")
        radar_values = df[radar_labels].mean().tolist()
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        radar_values += radar_values[:1]
        angles += angles[:1]
        fig2, ax2 = plt.subplots(subplot_kw={"polar": True})
        ax2.plot(angles, radar_values, "o-", linewidth=2)
        ax2.fill(angles, radar_values, alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_labels)
        ax2.set_title("Average Audio Profile")
        st.pyplot(fig2)

    # Tempo Distribution
    if "Tempo" in df.columns:
        st.subheader("\U0001F3B5 Tempo Distribution")
        fig3, ax3 = plt.subplots()
        ax3.hist(df["Tempo"].dropna(), bins=30)
        ax3.set_xlabel("Tempo (BPM)")
        ax3.set_ylabel("Track Count")
        ax3.set_title("Tempo Distribution")
        st.pyplot(fig3)

    # Loudness Distribution
    if "Loudness" in df.columns:
        st.subheader("\U0001F56A Loudness Distribution")
        fig4, ax4 = plt.subplots()
        ax4.hist(df["Loudness"].dropna(), bins=30, color="orange")
        ax4.set_xlabel("Loudness (dB)")
        ax4.set_ylabel("Track Count")
        ax4.set_title("Loudness Across Tracks")
        st.pyplot(fig4)

    # Word Cloud - Artists
    if "Artist Name(s)" in df.columns and df["Artist Name(s)"].notna().sum() > 0:
        st.subheader("\u2601\ufe0f Artist Word Cloud")
        artist_text = " ".join(df["Artist Name(s)"].dropna().astype(str).tolist())
        artist_wc = WordCloud(width=800, height=400, background_color="white").generate(artist_text)
        st.image(artist_wc.to_array(), use_container_width=True)
