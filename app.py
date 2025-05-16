import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="SonicMirror - Playlist Analyzer", layout="wide")
st.title("🎶 SonicMirror – Upload Your Spotify Playlists")

uploaded_files = st.file_uploader(
    "Upload one or more Exportify playlist files (CSV or Excel)", 
    type=["xlsx", "xls", "csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    all_dfs = []

    for file in uploaded_files:
        filename = file.name.rsplit(".", 1)[0]  # strip extension for playlist name

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            xls = pd.ExcelFile(file)
            df = pd.concat([xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True)

        df["Playlist"] = filename
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(subset=["Track Name", "Artist Name(s)"])
    df = df[df["Duration (ms)"] > 0]

    st.subheader("📋 Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df[["Track Name", "Artist Name(s)", "Playlist", "Release Date", "Popularity"]].head())

    # 🎛 Averages
    st.subheader("🎛 Key Audio Feature Averages")
    metrics = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness", "Tempo"]
    st.dataframe(df[metrics].mean().round(3).rename("Average").to_frame())

    # 🎨 Chart 1: Energy vs Valence
    st.subheader("🎨 Mood Map: Energy vs Valence")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Energy"], df["Valence"], alpha=0.5)
    ax1.set_xlabel("Energy")
    ax1.set_ylabel("Valence")
    ax1.set_title("Track Mood Distribution")
    st.pyplot(fig1)

    # 📅 Chart 2: Release Years by Decade
    st.subheader("📅 Songs by Decade")
    df["Release Year"] = pd.to_datetime(df["Release Date"], errors="coerce").dt.year
    df["Decade"] = (df["Release Year"] // 10 * 10).astype("Int64")
    decade_counts = df["Decade"].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(decade_counts.index.astype(str), decade_counts.values)
    ax2.set_title("Number of Songs by Decade")
    ax2.set_ylabel("Track Count")
    st.pyplot(fig2)

    # 🎤 Chart 3: Top 10 Artists
    st.subheader("🎤 Top 10 Artists")
    top_artists = df["Artist Name(s)"].value_counts().nlargest(10)
    fig3, ax3 = plt.subplots()
    top_artists.plot(kind="barh", ax=ax3)
    ax3.set_title("Most Frequent Artists")
    ax3.invert_yaxis()
    st.pyplot(fig3)

    # 🕸 Chart 4: Radar Chart of Audio Features
    st.subheader("🕸 Audio Feature Profile")
    radar_labels = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness"]
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

    # ⚖️ Chart 5: Speechiness vs Instrumentalness
    st.subheader("⚖️ Speechiness vs Instrumentalness")
    fig5, ax5 = plt.subplots()
    ax5.scatter(df["Speechiness"], df["Instrumentalness"], alpha=0.5)
    ax5.set_xlabel("Speechiness")
    ax5.set_ylabel("Instrumentalness")
    ax5.set_title("Vocals vs Instrumental")
    st.pyplot(fig5)

    # 🎵 Chart 6: Tempo Distribution
    st.subheader("🎵 Tempo Distribution")
    fig6, ax6 = plt.subplots()
    ax6.hist(df["Tempo"].dropna(), bins=30)
    ax6.set_xlabel("Tempo (BPM)")
    ax6.set_ylabel("Track Count")
    ax6.set_title("Tempo Distribution")
    st.pyplot(fig6)

    # 📣 Chart 7: Loudness Distribution
    if "Loudness" in df.columns:
        st.subheader("📣 Loudness Distribution")
        fig7, ax7 = plt.subplots()
        ax7.hist(df["Loudness"].dropna(), bins=30, color="orange")
        ax7.set_xlabel("Loudness (dB)")
        ax7.set_ylabel("Track Count")
        ax7.set_title("Loudness Across Tracks")
        st.pyplot(fig7)

    # 🎹 Chart 8: Most Common Keys
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
from wordcloud import WordCloud

# --- WORD CLOUDS: Artist + Genre ---
st.subheader("☁️ Artist & Genre Word Clouds")

if 'df' in locals() and "Artist Name(s)" in df.columns and df["Artist Name(s)"].notna().sum() > 0:
    artist_text = " ".join(df["Artist Name(s)"].dropna().astype(str).tolist())
    artist_wc = WordCloud(width=800, height=400, background_color="white").generate(artist_text)

    st.markdown("### 🎤 Most Frequent Artists")
    st.image(artist_wc.to_array(), use_container_width=True)
else:
    st.warning("No artist data found to generate word cloud.")

if 'df' in locals() and "Genres" in df.columns and df["Genres"].notna().sum() > 0:
    genre_text = " ".join(df["Genres"].dropna().astype(str).tolist())
    genre_wc = WordCloud(width=800, height=400, background_color="white").generate(genre_text)

    st.markdown("### 🎼 Most Common Genres")
    st.image(genre_wc.to_array(), use_container_width=True)
else:
    st.warning("No genre data found to generate word cloud.")


