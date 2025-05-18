import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ─── Page config ───
st.set_page_config(page_title="SonicMirror – Playlist Analyzer", layout="wide")

# ─── OAuth settings ───
CLIENT_ID     = st.secrets["SPOTIPY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]
REDIRECT_URI  = "https://sonicmirror.streamlit.app"
SCOPE         = "user-read-private playlist-read-private playlist-read-collaborative"

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    show_dialog=True
)

# ─── Authentication helper ───
def get_token():
    token_info = st.session_state.get("token_info") or sp_oauth.get_cached_token()
    params = st.query_params
    code_list = params.get("code")
    if code_list:
        code = code_list[0]
        raw = sp_oauth.get_access_token(code)
        token_info = raw if isinstance(raw, dict) else {"access_token": raw}
        st.session_state["token_info"] = token_info
        st.query_params = {}
        st.rerun()
    if token_info and sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        st.session_state["token_info"] = token_info
    return token_info

# ─── Main flow ───
token_info = get_token()
if not token_info:
    st.title("SonicMirror – Log in with Spotify")
    auth_url = sp_oauth.get_authorize_url()
    st.markdown(f"[🔐 Log in with Spotify]({auth_url})")
    st.stop()

sp = spotipy.Spotify(auth=token_info["access_token"])
user = sp.current_user()
st.sidebar.markdown(f"**Logged in as:** {user.get('display_name')} ({user.get('id')})")

# ─── Playlist selection ───
playlists = sp.current_user_playlists(limit=50).get('items', [])
options = {p['name']: p['id'] for p in playlists}
selected = st.multiselect("Select playlist(s)", list(options.keys()))

# ─── Fetch tracks and build DataFrame ───
if selected:
    all_tracks = []
    for name in selected:
        pid = options[name]
        tracks = []
        results = sp.playlist_items(pid, additional_types=['track'])
        while results:
            for item in results['items']:
                track = item.get('track')
                if track:
                    tracks.append(track)
            results = sp.next(results) if results.get('next') else None
        # Batch audio features (max 100 IDs per request)
        ids = [t['id'] for t in tracks if t.get('id')]
        features = {}
        for i in range(0, len(ids), 100):
            batch = ids[i:i+100]
            batch_feats = sp.audio_features(batch)
            for f in batch_feats or []:
                if f and f.get('id'):
                    features[f['id']] = f
        # Build rows
        for t in tracks:
            fid = t.get('id')
            af = features.get(fid, {})
            all_tracks.append({
                'Playlist': name,
                'Track Name': t.get('name'),
                'Artist': ', '.join([a['name'] for a in t.get('artists', [])]),
                'Energy': af.get('energy'),
                'Valence': af.get('valence'),
                'Danceability': af.get('danceability'),
                'Acousticness': af.get('acousticness'),
                'Instrumentalness': af.get('instrumentalness'),
                'Liveness': af.get('liveness'),
                'Tempo': af.get('tempo'),
                'Speechiness': af.get('speechiness'),
                'Loudness': af.get('loudness'),
                'Key': af.get('key')
            })
    df = pd.DataFrame(all_tracks)
    st.subheader("📋 Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df.head())
# 7) Handle Exportify file uploads
uploaded = st.file_uploader(
    "Upload Exportify playlist files (CSV/XLSX)", 
    type=["csv","xls","xlsx"], 
    accept_multiple_files=True
)
if uploaded:
    for f in uploaded:
        base = f.name.rsplit(".",1)[0]
        if f.name.endswith(".csv"):
            tmp = pd.read_csv(f)
        else:
            xls = pd.ExcelFile(f)
            tmp = pd.concat([xls.parse(s) for s in xls.sheet_names], ignore_index=True)
        tmp["Playlist"] = base
        st.session_state.setdefault("all_dfs", []).append(tmp)

# 8) Build the report when we have data
if st.session_state.get("all_dfs"):
    df = pd.concat(st.session_state["all_dfs"], ignore_index=True)
    df = df.dropna(subset=["Track Name","Artist Name(s)"])
    st.subheader("📋 Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df[["Track Name","Artist Name(s)","Playlist"]].head())

    # … your existing charts & tables here …


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
