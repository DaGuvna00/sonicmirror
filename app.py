import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

# ─── Page Config ───
st.set_page_config(page_title="SonicMirror Exportify Analyzer", layout="wide")
st.title("🎶 SonicMirror – Exportify Playlist Analyzer")

# ─── File Upload ───
st.sidebar.header("📂 Upload Exportify Exports")
uploaded_files = st.sidebar.file_uploader(
    "Select one or more Exportify CSV/Excel files", 
    type=["csv","xls","xlsx"],
    accept_multiple_files=True
)

# ─── Parse and Combine ┋
if uploaded_files:
    playlists = []
    for f in uploaded_files:
        name = f.name.rsplit('.',1)[0]
        try:
            if f.name.lower().endswith('.csv'):
                df = pd.read_csv(f)
            else:
                df = pd.concat(pd.read_excel(f, sheet_name=None).values(), ignore_index=True)
        except Exception as e:
            st.sidebar.error(f"Error reading {f.name}: {e}")
            continue
        # Normalize Exportify columns
        df = df.rename(columns={
            'Artist Name(s)': 'Artist',
            'Track Name': 'Track',
            'Added At': 'AddedAt',
            'Release Date': 'ReleaseDate'
        })
        df['Playlist'] = name
        playlists.append(df)
    if not playlists:
        st.error("No valid files uploaded.")
        st.stop()
    data = pd.concat(playlists, ignore_index=True)
else:
    st.info("Upload at least one Exportify file to get started.")
    st.stop()

# ─── Sidebar Controls ───
st.sidebar.header("🔎 Analysis Options")
playlist_names = data['Playlist'].unique().tolist()
selected = st.sidebar.multiselect("Choose playlists to include", playlist_names, default=playlist_names)

features = ['Energy','Valence','Danceability','Acousticness','Instrumentalness','Liveness','Speechiness','Tempo','Loudness']
selected_feats = st.sidebar.multiselect("Select audio features", features, default=features)

# Date parsing
data['AddedAt'] = pd.to_datetime(data['AddedAt'], errors='coerce')
data['ReleaseDate'] = pd.to_datetime(data['ReleaseDate'], errors='coerce')
# Discovery lag: days between release and add
data['LagDays'] = (data['AddedAt'] - data['ReleaseDate']).dt.days

# Filter data
df = data[data['Playlist'].isin(selected)].copy()

# ─── Main Dashboard ───
st.header("📋 Overview")
st.write(f"**Total tracks:** {len(df)} across {len(selected)} playlist(s)")

# Show raw table (first 10 rows)
st.subheader("Track Sample")
st.dataframe(df[['Playlist','Track','Artist','AddedAt','ReleaseDate'] + selected_feats].head(10))

# ─── Comparative Metrics ───
st.header("📊 Comparative Audio Feature Averages")
avgs = df.groupby('Playlist')[selected_feats].mean().round(3)
st.dataframe(avgs)

# Bar chart of a chosen feature
feat = st.selectbox("Feature to visualize by playlist", selected_feats)
fig, ax = plt.subplots()
avgs[feat].plot(kind='bar', ax=ax)
ax.set_ylabel(feat)
ax.set_title(f"Average {feat} by Playlist")
st.pyplot(fig)

# ─── Discovery Lag Distribution ───
st.header("⏱ Discovery Lag (Release → Add)")
fig2, ax2 = plt.subplots()
for pname in selected:
    subset = df[df['Playlist']==pname]
    ax2.hist(subset['LagDays'].dropna(), bins=30, alpha=0.5, label=pname)
ax2.set_xlabel('Days')
ax2.set_ylabel('Track Count')
ax2.legend()
ax2.set_title('Discovery Lag by Playlist')
st.pyplot(fig2)

# ─── Overlap & Uniques ───
st.header("🔗 Playlist Overlap & Unique Tracks")
sets = {p: set(df[df['Playlist']==p]['Track']) for p in selected}
# pairwise overlaps
if len(selected)>=2:
    import itertools
    ov_data = []
    for a,b in itertools.combinations(selected,2):
        count = len(sets[a].intersection(sets[b]))
        ov_data.append({'Pair':f"{a} & {b}", 'Overlap':count})
    ov_df = pd.DataFrame(ov_data)
    st.subheader("Pairwise Overlap Counts")
    st.bar_chart(ov_df.set_index('Pair'))
# unique counts
uniq = {p: len(sets[p] - set().union(*(sets[q] for q in selected if q!=p))) for p in selected}
uniq_df = pd.DataFrame.from_dict(uniq, orient='index', columns=['Unique Tracks'])
st.subheader("Unique Tracks per Playlist")
st.bar_chart(uniq_df)

# ─── Correlation Heatmap ───
st.header("🧩 Feature Correlation")
if selected_feats:
    corr = df[selected_feats].corr()
    fig3, ax3 = plt.subplots()
    cax = ax3.matshow(corr, vmin=-1, vmax=1)
    fig3.colorbar(cax)
    ax3.set_xticks(range(len(selected_feats)))
    ax3.set_yticks(range(len(selected_feats)))
    ax3.set_xticklabels(selected_feats, rotation=90)
    ax3.set_yticklabels(selected_feats)
    st.pyplot(fig3)

# ─── Word Clouds ───
st.header("☁️ Artist Word Cloud")
artist_text = ' '.join(df['Artist'].dropna().tolist())
if artist_text:
    wc = WordCloud(width=800, height=400, background_color='white').generate(artist_text)
    st.image(wc.to_array(), use_column_width=True)

# ─── Download Combined CSV ───
st.header("💾 Export Data")
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Download filtered data as CSV", buf.getvalue().encode('utf-8'), "filtered_playlists.csv")

# ─── Insights & Next Steps ───
st.header("📝 Insights & Ideas")
st.markdown("- Identify your most and least 'discovered' tracks by lag days.")
st.markdown("- Compare 'energy vs valence' scatter for each playlist.")
st.markdown("- Drill into seasonal trends by month or year added.")
st.markdown("- Add genre parsing to enrich word clouds.")


# ─── Render Report ───
if all_dfs:
    df = pd.concat(all_dfs, ignore_index=True)
    # Drop rows missing core columns
    df = df.dropna(subset=['Track Name', 'Artist'])
    st.subheader("📋 Combined Playlist Overview")
    st.write(f"**Total tracks:** {len(df)}")
    st.dataframe(df.head())

    # Download merged data
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    st.download_button("📥 Download combined CSV", buffer.getvalue().encode('utf-8'), 'combined.csv')

    # ─── Charting code goes here ───
    # (Mood maps, radar, histograms, word clouds, etc.)



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
