import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SonicMirror - Playlist Analyzer", layout="wide")
st.title("🎶 SonicMirror – Upload Your Spotify Playlists")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Exportify playlist file (CSV or Excel)", type=["xlsx", "xls", "csv"])

if uploaded_file:
    # --- Read the file ---
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        df["Playlist"] = "Uploaded CSV"
    else:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.concat([xls.parse(sheet).assign(Playlist=sheet) for sheet in xls.sheet_names], ignore_index=True)

    # --- Clean and prepare ---
    df = df.dropna(subset=["Track Name", "Artist Name(s)"])
    df = df[df["Duration (ms)"] > 0]  # skip broken rows

    st.subheader("📋 Playlist Overview")
    st.write(f"**Tracks loaded:** {len(df)}")
    st.dataframe(df[["Track Name", "Artist Name(s)", "Playlist", "Release Date", "Popularity"]].head())

    # --- Audio Summary Metrics ---
    st.subheader("🎛 Key Audio Averages")
    metrics = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness", "Tempo"]
    st.dataframe(df[metrics].mean().round(3).rename("Average").to_frame())

    # --- Chart: Energy vs Valence ---
    st.subheader("🎨 Mood Map: Energy vs Valence")
    fig, ax = plt.subplots()
    ax.scatter(df["Energy"], df["Valence"], alpha=0.5)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Valence")
    ax.set_title("Track Mood Distribution")
    st.pyplot(fig)

else:
    st.info("📁 Upload a playlist file to get started.")
