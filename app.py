import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="SonicMirror - Playlist Analyzer", layout="wide")
st.title("🎶 SonicMirror – Upload Your Spotify Playlists")

# --- File Upload ---
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

    # 🎨 Mood Map
    st.subheader("🎨 Mood Map: Energy vs Valence")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Energy"], df["Valence"], alpha=0.5)
    ax1.set_xlabel("Energy")
    ax1.set_ylabel("Valence")
    ax1.set_title("Track Mood Distribution")
    st.pyplot(fig1)

    # 📅 Songs by Decade
    st.subheader("📅 Songs by Decade")
    df["Release Year"] = pd.to_datetime(df["Release Date"], errors="coerce").dt.year
    df["Decade"] = (df["Release Year"] // 10 * 10).astype("Int64")
    decade_counts = df["Decade"].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(decade_counts.index.astype(str), decade_counts.values)
    ax2.set_title("Number of Songs by Decade")
    ax2.set_ylabel("Track Count")
    st.pyplot(fig2)

   # 📅 Time Between Track Release and When You Added It
st.subheader("📅 Time Between Track Release and When You Added It")

# Convert date columns
df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
df["Date Added"] = pd.to_datetime(df["Added At"], errors="coerce")

# Drop rows with missing dates
df = df.dropna(subset=["Release Date", "Date Added"])

# Calculate time difference
df["Days Until Added"] = (df["Date Added"] - df["Release Date"]).dt.days

# Plot
fig, ax = plt.subplots()
df["Days Until Added"].dropna().hist(bins=30, ax=ax, color="skyblue", edgecolor="black")
ax.set_title("Time Between Track Release and When You Added It")
ax.set_xlabel("Days Between Release and Add")
ax.set_ylabel("Number of Tracks")
st.pyplot(fig)


    # 🎤 Top Artists
st.subheader("🎤 Top 10 Artists")
top_artists = df["Artist Name(s)"].value_counts().nlargest(10)
fig3, ax3 = plt.subplots()
top_artists.plot(kind="barh", ax=ax3)
ax3.set_title("Most Frequent Artists")
ax3.invert_yaxis()
st.pyplot(fig3)

    # 🕸 Radar Chart of Audio Features
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

    # ⚖️ Speechiness vs Instrumentalness
st.subheader("⚖️ Speechiness vs Instrumentalness")
fig5, ax5 = plt.subplots()
ax5.scatter(df["Speechiness"], df["Instrumentalness"], alpha=0.5)
ax5.set_xlabel("Speechiness")
ax5.set_ylabel("Instrumentalness")
ax5.set_title("Vocals vs Instrumental")
st.pyplot(fig5)

    # 🎵 Tempo Distribution
    st.subheader("🎵 Tempo Distribution")
    fig6, ax6 = plt.subplots()
    ax6.hist(df["Tempo"].dropna(), bins=30)
    ax6.set_xlabel("Tempo (BPM)")
    ax6.set_ylabel("Track Count")
    ax6.set_title("Tempo Distribution")
    st.pyplot(fig6)

    # 📣 Loudness Distribution
    if "Loudness" in df.columns:
        st.subheader("📣 Loudness Distribution")
        fig7, ax7 = plt.subplots()
        ax7.hist(df["Loudness"].dropna(), bins=30, color="orange")
        ax7.set_xlabel("Loudness (dB)")
        ax7.set_ylabel("Track Count")
        ax7.set_title("Loudness Across Tracks")
        st.pyplot(fig7)

    # 🎹 Most Common Keys
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

    # 🆚 Playlist Comparison (Radar + Bar + Mood Map)
    st.subheader("📊 Compare Playlists Side-by-Side")
    playlist_names = df["Playlist"].unique()
    selected = st.multiselect("Choose playlists to compare", playlist_names, default=list(playlist_names))

    if selected:
        features = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Liveness"]
        avg_df = df[df["Playlist"].isin(selected)].groupby("Playlist")[features].mean()

        # Radar
        labels = features
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        fig_radar, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(6, 6))
        for playlist in avg_df.index:
            values = avg_df.loc[playlist].tolist()
            values += values[:1]
            ax.plot(angles, values, label=playlist)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title("Playlist Audio Profiles")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig_radar)

        # Mood Scatter
        fig_mood, ax2 = plt.subplots()
        for playlist in selected:
            pl_df = df[df["Playlist"] == playlist]
            ax2.scatter(pl_df["Energy"], pl_df["Valence"], alpha=0.4, label=playlist)
        ax2.set_xlabel("Energy")
        ax2.set_ylabel("Valence")
        ax2.set_title("Mood Distribution by Playlist")
        ax2.legend()
        st.pyplot(fig_mood)

        # Bar Chart
        st.subheader("📶 Average Features Per Playlist")
        st.dataframe(avg_df.round(3).transpose())
        avg_df_plot = avg_df.transpose()
        fig_bar, ax3 = plt.subplots(figsize=(10, 5))
        avg_df_plot.plot(kind="bar", ax=ax3)
        ax3.set_title("Average Audio Features by Playlist")
        ax3.set_ylabel("Value")
        ax3.set_xticklabels(avg_df_plot.index, rotation=0)
        st.pyplot(fig_bar)

    # ☁️ Word Clouds
    st.subheader("☁️ Artist & Genre Word Clouds")

    if "Artist Name(s)" in df.columns and df["Artist Name(s)"].notna().sum() > 0:
        artist_text = " ".join(df["Artist Name(s)"].dropna().astype(str).tolist())
        artist_wc = WordCloud(width=800, height=400, background_color="white").generate(artist_text)
        st.markdown("### 🎤 Most Frequent Artists")
        st.image(artist_wc.to_array(), use_container_width=True)
    else:
        st.warning("No artist data found to generate word cloud.")

    if "Genres" in df.columns and df["Genres"].notna().sum() > 0:
        genre_text = " ".join(df["Genres"].dropna().astype(str).tolist())
        genre_wc = WordCloud(width=800, height=400, background_color="white").generate(genre_text)
        st.markdown("### 🎼 Most Common Genres")
        st.image(genre_wc.to_array(), use_container_width=True)
    else:
        st.warning("No genre data found to generate word cloud.")

st.subheader("🧠 Playlist Personality Summary")

# Features for mood insight
summary_features = ["Valence", "Energy", "Danceability", "Acousticness", "Speechiness", "Tempo"]
personality = df[summary_features].mean()

def interpret_personality(p):
    traits = []

    if p["Valence"] > 0.6:
        traits.append("😊 Positive")
    elif p["Valence"] < 0.4:
        traits.append("😔 Moody")

    if p["Energy"] > 0.7:
        traits.append("⚡ High Energy")
    elif p["Energy"] < 0.4:
        traits.append("🧘 Calm")

    if p["Danceability"] > 0.7:
        traits.append("💃 Danceable")
    elif p["Danceability"] < 0.4:
        traits.append("🪑 Chill")

    if p["Acousticness"] > 0.5:
        traits.append("🌿 Organic")
    else:
        traits.append("🎛 Electronic")

    if p["Speechiness"] > 0.33:
        traits.append("🗣 Spoken or Rap-heavy")

    if p["Tempo"] > 120:
        traits.append("🏃 Fast-paced")
    elif p["Tempo"] < 90:
        traits.append("🚶 Slow & Steady")

    return traits

traits = interpret_personality(personality)

st.markdown("### 🪞 This Playlist Feels Like:")
st.markdown("**" + ", ".join(traits) + "**")

# Optional: add emoji summary radar or text bar
st.markdown("> _Based on your playlist’s audio features._")
