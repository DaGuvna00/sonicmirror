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
all_data = []

# Helper to chunk track IDs

def chunked(iterable, size=100):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

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

        selected = st.multiselect("\U0001F3A7 Choose One or More Playlists", playlist_names)

        for sel in selected:
            idx = playlist_names.index(sel)
            playlist_id = playlist_ids[idx]
            tracks_data = sp.playlist_tracks(playlist_id)

            track_ids = []
            track_names = []
            artists = []
            albums = []

            for item in tracks_data['items']:
                track = item['track']
                if track and track['id']:
                    track_ids.append(track['id'])
                    track_names.append(track['name'])
                    artists.append(", ".join([a['name'] for a in track['artists']]))
                    albums.append(track['album']['name'])
        if not track_ids:
            st.warning(f"⚠️ No valid track IDs found for playlist '{sel}'. Skipping.")
        else:
        
        features = []
valid_track_ids = [tid for tid in track_ids if tid]

for chunk in chunked(valid_track_ids):
    try:
        data = sp.audio_features(chunk)
        if data:
            clean_data = [f for f in data if f is not None]
            if clean_data:
                features.extend(clean_data)
    except spotipy.SpotifyException as e:
        st.warning(f"Skipped a chunk due to Spotify error: {e}")

# Only proceed if we got audio features
if features:
    df = pd.DataFrame(features)
    df["Track Name"] = track_names[:len(df)]
    df["Artist Name(s)"] = artists[:len(df)]
    df["Album Name"] = albums[:len(df)]
    df["Playlist"] = sel

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

    all_data.append(df)
else:
    st.warning(f"⚠️ No audio features found for playlist '{sel}'. It may contain unavailable tracks or Spotify may have blocked the request.")


# Combine all collected data
if all_data:
    df = pd.concat(all_data, ignore_index=True)

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
if 'df' in locals() and df is not None:
    st.subheader("\U0001F4CB Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df.head())

    # Comparison: Radar per playlist
    st.subheader("\U0001F9EA Playlist Comparison – Radar Chart")
    metrics = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness"]
    available_metrics = [m for m in metrics if m in df.columns]
if not available_metrics:
    st.warning("No audio features available for comparison.")
else:
    grouped = df.groupby("Playlist")[available_metrics].mean()
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    for playlist in grouped.index:
        values = grouped.loc[playlist].tolist()
        values += values[:1]
        ax.plot(angles, values, label=playlist)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics)
    ax.set_title("Average Audio Features by Playlist")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    st.pyplot(fig)




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

    # Key Distribution
    if "Key" in df.columns:
        st.subheader("\U0001F3B9 Most Common Musical Keys")
        key_names = {
            0: "C", 1: "C♯/D♭", 2: "D", 3: "D♯/E♭", 4: "E", 5: "F",
            6: "F♯/G♭", 7: "G", 8: "G♯/A♭", 9: "A", 10: "A♯/B♭", 11: "B"
        }
        key_counts = df["Key"].map(key_names).value_counts().sort_index()
        fig5, ax5 = plt.subplots()
        key_counts.plot(kind="bar", ax=ax5)
        ax5.set_title("Most Common Musical Keys")
        ax5.set_xlabel("Key")
        ax5.set_ylabel("Track Count")
        st.pyplot(fig5)

    # Word Cloud - Artists
    if "Artist Name(s)" in df.columns and df["Artist Name(s)"].notna().sum() > 0:
        st.subheader("\u2601\ufe0f Artist Word Cloud")
        artist_text = " ".join(df["Artist Name(s)"].dropna().astype(str).tolist())
        artist_wc = WordCloud(width=800, height=400, background_color="white").generate(artist_text)
        st.image(artist_wc.to_array(), use_container_width=True)

    # Word Cloud - Albums
    if "Album Name" in df.columns and df["Album Name"].notna().sum() > 0:
        st.subheader("\u2601\ufe0f Album Word Cloud")
        album_text = " ".join(df["Album Name"].dropna().astype(str).tolist())
        album_wc = WordCloud(width=800, height=400, background_color="white").generate(album_text)
        st.image(album_wc.to_array(), use_container_width=True)
