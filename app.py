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
st.title("🎶 SonicMirror – Analyze Your Spotify Playlists")

auth_url = sp_oauth.get_authorize_url()
st.markdown(f"[🔐 Log in with Spotify]({auth_url})")

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

        selected = st.selectbox("🎧 Choose a Playlist", playlist_names)

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

            features = []
            valid_track_ids = [tid for tid in track_ids if tid]

            for i in range(0, len(valid_track_ids), 100):
                chunk = valid_track_ids[i:i + 100]
                try:
                    chunk_features = sp.audio_features(chunk)
                    if chunk_features:
                        clean_chunk = [f for f in chunk_features if f is not None]
                        features.extend(clean_chunk)
                except spotipy.SpotifyException as e:
                    st.warning(f"⚠️ Skipped a chunk due to Spotify error: {e}")

            if not features:
                st.error("❌ No audio features could be retrieved. Tracks may be unavailable.")
                st.stop()

            df = pd.DataFrame(features)
            df["Track Name"] = track_names[:len(df)]
            df["Artist Name(s)"] = artists[:len(df)]

            st.subheader("📋 Playlist Tracks with Features")
            st.write(f"**Total Tracks:** {len(df)}")
            st.dataframe(df[[
                "Track Name", "Artist Name(s)", "energy", "valence", "danceability", "acousticness", "tempo"
            ]])

            st.subheader("🎛 Key Audio Feature Averages")
            st.dataframe(df[["energy", "valence", "danceability", "acousticness", "tempo"]].mean().round(3).rename("Average").to_frame())

            st.subheader("🎨 Mood Map: Energy vs Valence")
            fig, ax = plt.subplots()
            ax.scatter(df["energy"], df["valence"], alpha=0.5)
            ax.set_xlabel("Energy")
            ax.set_ylabel("Valence")
            ax.set_title("Track Mood Distribution")
            st.pyplot(fig)

            st.subheader("☁️ Artist Word Cloud")
            if "Artist Name(s)" in df.columns and df["Artist Name(s)"].notna().sum() > 0:
                artist_text = " ".join(df["Artist Name(s)"].dropna().astype(str).tolist())
                artist_wc = WordCloud(width=800, height=400, background_color="white").generate(artist_text)
                st.image(artist_wc.to_array(), use_container_width=True)
