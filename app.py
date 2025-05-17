import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Streamlit Setup ---
st.set_page_config(page_title="SonicMirror - Playlist Analyzer", layout="wide")
st.title("🎶 SonicMirror – Analyze Your Spotify Playlists")

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

auth_url = sp_oauth.get_authorize_url()
st.markdown(f"[🔐 Log in with Spotify]({auth_url})")

# --- Spotify OAuth Token Management ---
all_dfs = []

# Get code from URL
query_params = st.query_params
code = query_params.get("code", [None])[0]

if code and "token_info" not in st.session_state:
    st.session_state["auth_code"] = code  # Save code quickly to avoid reuse timeout

if "auth_code" in st.session_state and "token_info" not in st.session_state:
    try:
        token_info = sp_oauth.get_access_token(st.session_state["auth_code"], as_dict=True)
        if token_info and token_info.get("access_token"):
            st.session_state.token_info = token_info
        else:
            st.error("Spotify login failed: access token missing.")
    except Exception as e:
        st.error("Spotify OAuth error during token exchange.")
        st.exception(e)
        st.stop()

if "token_info" in st.session_state:
    if sp_oauth.is_token_expired(st.session_state.token_info):
        try:
            st.session_state.token_info = sp_oauth.refresh_access_token(st.session_state.token_info['refresh_token'])
        except Exception as e:
            st.error("Could not refresh Spotify token. Please log in again.")
            st.exception(e)
            st.stop()

    access_token = st.session_state.token_info.get("access_token")
    st.write("Access token preview:", access_token[:10] + "...")  # Debug

    # --- Manual token test ---
    st.write("🔍 Access token headers test:")
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        test_url = "https://api.spotify.com/v1/me"
        resp = requests.get(test_url, headers=headers)
        st.code(resp.status_code)
        st.json(resp.json())
    except Exception as e:
        st.error("Manual token test failed:")
        st.exception(e)

    if access_token:
        sp = spotipy.Spotify(auth=access_token)

        try:
            user = sp.current_user()
            st.success(f"Logged in as {user['display_name']}")
        except spotipy.exceptions.SpotifyException:
            st.error("Failed to fetch user profile. Your token might have expired.")

        # --- Playlist Selection ---
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
                if track and track['id'] and track.get('is_playable', True) and track['type'] == 'track':
                    # st.json(track)  # Optional debug
                    track_ids.append(track['id'])
                    track_names.append(track['name'])
                    artists.append(", ".join([a['name'] for a in track['artists']]))

            st.write("Track IDs for audio_features:", track_ids)  # Debug

            if track_ids:
                try:
                    audio_features = sp.audio_features(track_ids)
                    st.write("Audio features fetched successfully.")  # Debug
                    if audio_features:
                        df = pd.DataFrame(audio_features)
                        df["Track Name"] = track_names
                        df["Artist Name(s)"] = artists
                        df["Playlist"] = selected
                        all_dfs.append(df)
                    else:
                        st.error("Received empty audio features list.")
                except Exception as e:
                    st.error("Error fetching audio features:")
                    st.exception(e)
                    st.stop()
    else:
        st.error("Access token is missing or invalid. Please re-authenticate.")

# --- Exportify Upload ---
uploaded_files = st.file_uploader(
    "Upload one or more Exportify playlist files (CSV or Excel)",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        filename = file.name.rsplit(".", 1)[0]
        if file.name.endswith(".csv"):
            temp_df = pd.read_csv(file)
        else:
            xls = pd.ExcelFile(file)
            temp_df = pd.concat([xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True)

        temp_df["Playlist"] = filename
        all_dfs.append(temp_df)

# --- Chart + Report Section ---
if all_dfs:
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(subset=["Track Name", "Artist Name(s)"])
    if "Duration (ms)" in df.columns:
        df = df[df["Duration (ms)"] > 0]

    st.subheader("📋 Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df[["Track Name", "Artist Name(s)", "Playlist"] +
                    [col for col in ["Release Date", "Popularity"] if col in df.columns]].head())

    # 🎛 Averages
    st.subheader("🎛 Key Audio Feature Averages")
    metrics = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness", "Tempo"]
    st.dataframe(df[[m for m in metrics if m in df.columns]].mean().round(3).rename("Average").to_frame())

    # 🎨 Mood Map
    if "Energy" in df.columns and "Valence" in df.columns:
        st.subheader("🎨 Mood Map: Energy vs Valence")
        fig1, ax1 = plt.subplots()
        ax1.scatter(df["Energy"], df["Valence"], alpha=0.5)
        ax1.set_xlabel("Energy")
        ax1.set_ylabel("Valence")
        ax1.set_title("Track Mood Distribution")
        st.pyplot(fig1)

    # 📅 Songs by Decade
    if "Release Date" in df.columns:
        st.subheader("📅 Songs by Decade")
        df["Release Year"] = pd.to_datetime(df["Release Date"], errors="coerce").dt.year
        df["Decade"] = (df["Release Year"] // 10 * 10).astype("Int64")
        decade_counts = df["Decade"].value_counts().sort_index()
        fig2, ax2 = plt.subplots()
        ax2.bar(decade_counts.index.astype(str), decade_counts.values)
        ax2.set_title("Number of Songs by Decade")
        ax2.set_ylabel("Track Count")
        st.pyplot(fig2)

    # 🎤 Top Artists
    st.subheader("🎤 Top 10 Artists")
    top_artists = df["Artist Name(s)"].value_counts().nlargest(10)
    fig3, ax3 = plt.subplots()
    top_artists.plot(kind="barh", ax=ax3)
    ax3.set_title("Most Frequent Artists")
    ax3.invert_yaxis()
    st.pyplot(fig3)

    # 🕸 Radar Chart
    st.subheader("🕸 Audio Feature Profile")
    radar_labels = [m for m in metrics if m in df.columns]
    if radar_labels:
        radar_values = df[radar_labels].mean().tolist()
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        radar_values += radar_values[:1]
        angles += angles[:1]
        fig4, ax4 = plt.subplots(subplot_kw={"polar": True})
        ax4.plot(angles, radar_values, "o-", linewidth=2)
        ax4.fill(angles, radar_values, alpha=0.25)
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(radar_labels)
        ax4.set_title("Average Audio Profile")
        st.pyplot(fig4)

    # ⚖️ Speechiness vs Instrumentalness
    if "Speechiness" in df.columns and "Instrumentalness" in df.columns:
        st.subheader("⚖️ Speechiness vs Instrumentalness")
        fig5, ax5 = plt.subplots()
        ax5.scatter(df["Speechiness"], df["Instrumentalness"], alpha=0.5)
        ax5.set_xlabel("Speechiness")
        ax5.set_ylabel("Instrumentalness")
        ax5.set_title("Vocals vs Instrumental")
        st.pyplot(fig5)

    # 🎵 Tempo Distribution
    if "Tempo" in df.columns:
        st.subheader("🎵 Tempo Distribution")
        fig6, ax6 = plt.subplots()
        ax6.hist(df["Tempo"].dropna(), bins=30)
        ax6.set_xlabel("Tempo (BPM)")
        ax6.set_ylabel("Track Count")
        ax6.set_title("Tempo Distribution")
        st.pyplot(fig6)

    # 📣 Loudness
    if "Loudness" in df.columns:
        st.subheader("📣 Loudness Distribution")
        fig7, ax7 = plt.subplots()
        ax7.hist(df["Loudness"].dropna(), bins=30, color="orange")
        ax7.set_xlabel("Loudness (dB)")
        ax7.set_ylabel("Track Count")
        ax7.set_title("Loudness Across Tracks")
        st.pyplot(fig7)

    # 🎹 Key Distribution
    if "Key" in df.columns:
        st.subheader("🎹 Most Common Musical Keys")
        key_names = {
            0: "C", 1: "C♯/D♭", 2: "D", 3: "D♯/E♭", 4: "E", 5: "F",
            6: "F♯/G♭", 7: "G", 8: "G♯/A♭", 9: "A", 10: "A♯/B♭", 11: "B"
        }
        key_counts = df["Key"].map(key_names).value_counts().sort_index()
        fig8, ax8 = plt.subplots()
        key_counts.plot(kind="bar", ax=ax8)
        ax8.set_title("Most Common Musical Keys")
        ax8.set_xlabel("Key")
        ax8.set_ylabel("Track Count")
        st.pyplot(fig8)

    # ☁️ Word Clouds
    st.subheader("☁️ Artist & Genre Word Clouds")
    if "Artist Name(s)" in df.columns:
        artist_text = " ".join(df["Artist Name(s)"].dropna())
        artist_wc = WordCloud(width=800, height=400, background_color="white").generate(artist_text)
        st.markdown("### 🎤 Most Frequent Artists")
        st.image(artist_wc.to_array(), use_container_width=True)

    if "Genres" in df.columns:
        genre_text = " ".join(df["Genres"].dropna().astype(str))
        genre_wc = WordCloud(width=800, height=400, background_color="white").generate(genre_text)
        st.markdown("### 🎼 Most Common Genres")
        st.image(genre_wc.to_array(), use_container_width=True)
