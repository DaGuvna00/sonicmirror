import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spotipy
from spotipy.oauth2 import SpotifyOAuth

st.set_page_config(page_title="SonicMirror - Playlist Analyzer", layout="wide")
st.title("🎶 SonicMirror – Analyze Your Spotify Playlists")

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

auth_url = sp_oauth.get_authorize_url()
st.markdown(f"[🔐 Log in with Spotify]({auth_url})")

def chunked(iterable, size=100):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

all_data = []
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

        selected = st.multiselect("🎧 Choose One or More Playlists", playlist_names)

        for sel in selected:
            idx = playlist_names.index(sel)
            playlist_id = playlist_ids[idx]
            tracks_data = sp.playlist_tracks(playlist_id)

            track_ids, track_names, artists, albums = [], [], [], []

            for item in tracks_data['items']:
                track = item['track']
                if track and track['id']:
                    track_ids.append(track['id'])
                    track_names.append(track['name'])
                    artists.append(", ".join([a['name'] for a in track['artists']]))
                    albums.append(track['album']['name'])

            if not track_ids:
                st.warning(f"⚠️ No valid track IDs found for playlist '{sel}'. Skipping.")
                continue

            features = []
            for chunk in chunked(track_ids):
                try:
                    audio_data = sp.audio_features(chunk)
                    if audio_data:
                        features.extend([f for f in audio_data if f])
                except spotipy.SpotifyException as e:
                    st.warning(f"Spotify error: {e}")

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
                st.warning(f"⚠️ No audio features found for '{sel}'.")

# Combine all collected data
if all_data:
    df = pd.concat(all_data, ignore_index=True)

# Charts
if 'df' in locals() and not df.empty:
    st.subheader("📋 Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df.head())

    # Radar
    st.subheader("🧠 Playlist Comparison – Radar Chart")
    metrics = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness"]
    available = [m for m in metrics if m in df.columns]

    if not available:
        st.warning("No audio features available.")
    else:
        grouped = df.groupby("Playlist")[available].mean()
        angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(subplot_kw={"polar": True})
        for playlist in grouped.index:
            values = grouped.loc[playlist].tolist() + grouped.loc[playlist].tolist()[:1]
            ax.plot(angles, values, label=playlist)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available)
        ax.set_title("Average Audio Features by Playlist")
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        st.pyplot(fig)

    # Tempo Histogram
    if "Tempo" in df.columns:
        st.subheader("🎵 Tempo Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Tempo"].dropna(), bins=30)
        ax.set_xlabel("Tempo (BPM)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Loudness Histogram
    if "Loudness" in df.columns:
        st.subheader("📣 Loudness")
        fig, ax = plt.subplots()
        ax.hist(df["Loudness"].dropna(), bins=30)
        ax.set_xlabel("Loudness (dB)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Word Clouds
    if "Artist Name(s)" in df.columns:
        st.subheader("☁️ Artist Word Cloud")
        artist_text = " ".join(df["Artist Name(s)"].dropna().astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(artist_text)
        st.image(wc.to_array(), use_container_width=True)

    if "Album Name" in df.columns:
        st.subheader("☁️ Album Word Cloud")
        album_text = " ".join(df["Album Name"].dropna().astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(album_text)
        st.image(wc.to_array(), use_container_width=True)
