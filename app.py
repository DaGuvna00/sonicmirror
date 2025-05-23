import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import seaborn as sns

# ─── Page Config ───
st.set_page_config(page_title="SonicMirror Exportify Analyzer", layout="wide")
st.title("🎶 SonicMirror – Exportify Playlist Analyzer")


# ─── Upload and Parse Exportify Files ───
st.sidebar.header("📂 Upload Exportify Exports")

uploaded_files = st.sidebar.file_uploader(
    "Select one or more Exportify CSV/Excel files",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True,
    key="file_uploader"
)

# Store uploaded files in session_state
if uploaded_files:
    st.session_state['raw_files'] = uploaded_files

# Trigger parse only after user confirms
if 'raw_files' in st.session_state and st.session_state['raw_files']:
    st.success(f"{len(st.session_state['raw_files'])} file(s) uploaded.")

    if st.button("🔍 Start Analysis") or st.session_state.get("retry_failed"):
        import time
        import io
        playlists = []
        failed_files = []
        status_messages = []

        with st.spinner("📱 Processing your playlists..."):
            time.sleep(0.3)

            for f in st.session_state['raw_files']:
                name = f.name.rsplit('.', 1)[0]
                df = None

                for attempt in range(3):
                    try:
                        f.seek(0)
                        buffer = io.BytesIO(f.read())
                        buffer.seek(0)

                        if f.name.lower().endswith('.csv'):
                            df = pd.read_csv(buffer, encoding='utf-8')
                        else:
                            buffer.seek(0)
                            sheets = pd.read_excel(buffer, sheet_name=None)
                            df = pd.concat(sheets.values(), ignore_index=True)

                        if df.empty or df.shape[1] < 2:
                            raise ValueError("Empty or invalid structure")

                        break  # successful read
                    except Exception:
                        df = None
                        time.sleep(0.2)

                if df is None:
                    failed_files.append(f.name)
                    status_messages.append(f"❌ {f.name} — Failed after 3 attempts")
                    continue

                df = df.rename(columns={
                    'Artist Name(s)': 'Artist',
                    'Track Name': 'Track',
                    'Added At': 'AddedAt',
                    'Release Date': 'ReleaseDate'
                })

                df['Playlist'] = name
                playlists.append(df)
                status_messages.append(f"✅ {f.name} — Loaded successfully")

        for msg in status_messages:
            st.markdown(msg)

        if not playlists:
            st.error("⚠️ No valid playlists found. Try uploading again.")
            st.session_state.pop("retry_failed", None)
            st.stop()

        data = pd.concat(playlists, ignore_index=True)
        st.session_state['data'] = data
        st.session_state.pop("retry_failed", None)
        st.success("✅ Playlists loaded successfully. Ready to analyze!")

        # Retry failed button
        if failed_files:
            if st.button("🔁 Retry Failed Files"):
                st.session_state['raw_files'] = [f for f in uploaded_files if f.name in failed_files]
                st.session_state['retry_failed'] = True
                st.experimental_rerun()

# Only show analysis controls if data is parsed
if 'data' in st.session_state:
    data = st.session_state['data']
    st.sidebar.header("🔎 Analysis Options")
    playlist_names = data['Playlist'].unique().tolist()
    selected = st.sidebar.multiselect("Choose playlists to include", playlist_names, default=playlist_names)
    features = ['Energy','Valence','Danceability','Acousticness','Instrumentalness','Liveness','Speechiness','Tempo','Loudness']
    selected_feats = st.sidebar.multiselect("Select audio features", features, default=features)
else:
    st.sidebar.info("Upload playlists and click 'Start Analysis' to begin.")
# ─── Date Parsing & Lag Calculation ───
if 'data' in st.session_state:
    data = st.session_state['data']
    
    # Convert to UTC then drop timezone so both columns align
    data['AddedAt'] = pd.to_datetime(data['AddedAt'], errors='coerce', utc=True).dt.tz_convert(None)
    data['ReleaseDate'] = pd.to_datetime(data['ReleaseDate'], errors='coerce', utc=True).dt.tz_convert(None)

    # Compute discovery lag in days
    data['LagDays'] = (data['AddedAt'] - data['ReleaseDate']).dt.days



# ─── Filter & Prepare Dashboard Data ───
if 'data' in st.session_state:
    data = st.session_state['data']
    df = data[data['Playlist'].isin(selected)].copy()
else:
    st.warning("No playlist data available yet. Upload and run analysis first.")
    st.stop()

# ─── Main Dashboard ───
st.header("📋 Combined Playlist Overview")
st.write(f"**Total Tracks:** {len(df)} across {len(selected)} playlist(s)")

# Sample of raw data
st.subheader("🔍 Data Sample")
st.dataframe(df[['Playlist','Track','Artist','AddedAt','ReleaseDate','LagDays'] + selected_feats].head(10))

# ─── Comparative Feature Averages ───
st.header("📊 Average Audio Features by Playlist")
avgs = df.groupby('Playlist')[selected_feats].mean().round(3)
st.dataframe(avgs)

# Visual: selected feature
feat = st.selectbox("Visualize feature", selected_feats)
fig, ax = plt.subplots()
avgs[feat].plot(kind='bar', ax=ax)
ax.set_ylabel(feat)
ax.set_title(f"Average {feat} by Playlist")
st.pyplot(fig)

# ─── Discovery Lag Distribution ───
st.header("⏱ Discovery Lag Distribution")
fig2, ax2 = plt.subplots()
for p in selected:
    subset = df[df['Playlist']==p]
    ax2.hist(subset['LagDays'].dropna(), bins=30, alpha=0.5, label=p)
ax2.set_xlabel('Lag (Days)')
ax2.set_ylabel('Track Count')
ax2.legend()
st.pyplot(fig2)

# ─── Playlist Mood Ring Comparison ───
st.header("🌈 Playlist Mood Ring")

# Dropdowns to select playlists
p1 = st.selectbox("Select Playlist 1", df['Playlist'].unique(), key="mood1")
p2 = st.selectbox("Select Playlist 2", df['Playlist'].unique(), key="mood2")

# Define features and corresponding emojis
mood_features = ['Energy', 'Valence', 'Danceability', 'Liveness', 'Speechiness', 'Acousticness']
mood_emojis = {
    'Energy': '⚡', 'Valence': '😊', 'Danceability': '🕺',
    'Liveness': '🎤', 'Speechiness': '🗣️', 'Acousticness': '🌿'
}

if p1 and p2:
    angles = np.linspace(0, 2 * np.pi, len(mood_features), endpoint=False).tolist()
    angles += angles[:1]

    # Compute averages and close the radar loop
    avg1 = df[df['Playlist'] == p1][mood_features].mean().tolist()
    avg2 = df[df['Playlist'] == p2][mood_features].mean().tolist()
    avg1 += avg1[:1]
    avg2 += avg2[:1]

    # Determine dominant feature for emoji
    dominant_idx = np.argmax(df[df['Playlist'] == p1][mood_features].mean())
    emoji = mood_emojis[mood_features[dominant_idx]]

    # Radar plot
    fig_mood, ax_mood = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_mood.plot(angles, avg1, color='orchid', linewidth=2, label=p1)
    ax_mood.fill(angles, avg1, color='orchid', alpha=0.3)
    ax_mood.plot(angles, avg2, color='darkorange', linewidth=2, label=p2)
    ax_mood.fill(angles, avg2, color='darkorange', alpha=0.3)
    ax_mood.set_xticks(angles[:-1])
    ax_mood.set_xticklabels(mood_features)
    ax_mood.set_title("🎭 Audio Mood Radar")

    # Add center emoji
    ax_mood.text(0, 0, emoji, fontsize=38, ha='center', va='center')

    ax_mood.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig_mood)



# ─── Overlap & Unique Tracks ───
st.header("🔗 Playlist Overlap & Unique")
sets = {p: set(df[df['Playlist']==p]['Track']) for p in selected}
import itertools
# Pairwise overlap counts
ov_data = []
for a, b in itertools.combinations(selected, 2):
    ov = len(sets[a] & sets[b])
    ov_data.append({'Pair': f"{a} & {b}", 'Overlap': ov})
ov_df = pd.DataFrame(ov_data)
st.subheader("Pairwise Overlap")
st.bar_chart(ov_df.set_index('Pair'))
# Unique per playlist
uniq = {p: len(sets[p] - set().union(*(sets[q] for q in selected if q!=p))) for p in selected}
uniq_df = pd.DataFrame.from_dict(uniq, orient='index', columns=['UniqueTracks'])
st.subheader("Unique Tracks per Playlist")
st.bar_chart(uniq_df)

# ─── Correlation Matrix ───
st.header("🧩 Feature Correlation")
corr = df[selected_feats].corr()
fig3, ax3 = plt.subplots()
cax = ax3.matshow(corr, vmin=-1, vmax=1)
fig3.colorbar(cax)
ax3.set_xticks(range(len(selected_feats)))
ax3.set_yticks(range(len(selected_feats)))
ax3.set_xticklabels(selected_feats, rotation=90)
ax3.set_yticklabels(selected_feats)
st.pyplot(fig3)

# ─── Word Cloud ───
st.header("☁️ Artist Word Cloud")
artist_text = ' '.join(df['Artist'].dropna().tolist())
if artist_text:
    wc = WordCloud(width=800, height=400, background_color='white').generate(artist_text)
    st.image(wc.to_array(), use_container_width=True)

# ─── Genre Word Cloud ───
if "Genres" in df.columns:
    st.header("☁️ Genre Word Cloud")
    genre_text = ' '.join(df['Genres'].dropna().astype(str).tolist())
    if genre_text:
        genre_wc = WordCloud(width=800, height=400, background_color='white').generate(genre_text)
        st.image(genre_wc.to_array(), use_container_width=True)
    else:
        st.info("No genre data available to generate word cloud.")
else:
    st.info("Genres column not found in data.")


# ─── Export Filtered Data ───
st.header("💾 Download Data")
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Download CSV", buf.getvalue().encode('utf-8'), 'sonicmirror_export.csv')

# ─── Valence vs Energy Scatter Plot ───
st.header("🎨 Valence vs Energy: Mood Mapping")
fig4, ax4 = plt.subplots()
for p in selected:
    subset = df[df['Playlist']==p]
    ax4.scatter(
        subset['Valence'], subset['Energy'],
        alpha=0.6, label=p, s=40
    )
ax4.set_xlabel('Valence')
ax4.set_ylabel('Energy')
ax4.set_title('Track Mood Distribution by Playlist')
ax4.legend()
st.pyplot(fig4)

# ─── Time-Series of Tracks Added ───
st.header("📈 Tracks Added Over Time")
# Group by month-year
df['Month'] = df['AddedAt'].dt.to_period('M').dt.to_timestamp()
time_data = df.groupby(['Month','Playlist']).size().reset_index(name='Count')
fig5, ax5 = plt.subplots(figsize=(10,4))
for p in selected:
    series = time_data[time_data['Playlist']==p]
    ax5.plot(series['Month'], series['Count'], marker='o', label=p)
ax5.set_xlabel('Month')
ax5.set_ylabel('Number of Tracks Added')
ax5.set_title('Tracks Added Over Time by Playlist')
ax5.legend()
st.pyplot(fig5)

# ─── Seasonal Trend Analysis ───
st.header("🌦 Seasonal Additions by Month")
# Extract month name
df['MonthName'] = df['AddedAt'].dt.month_name()
season_data = df.groupby(['MonthName','Playlist']).size().unstack(fill_value=0)
# Ensure month order
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
season_data = season_data.reindex(months)
st.dataframe(season_data)

# ─── Valence vs Energy Scatter Plot ───
st.header("🎨 Valence vs Energy: Mood Mapping")
fig4, ax4 = plt.subplots()
for p in selected:
    subset = df[df['Playlist']==p]
    ax4.scatter(
        subset['Valence'], subset['Energy'],
        alpha=0.6, label=p, s=40
    )
ax4.set_xlabel('Valence')
ax4.set_ylabel('Energy')
ax4.set_title('Track Mood Distribution by Playlist')
ax4.legend()
st.pyplot(fig4)

# ─── Genre & Subgenre Breakdown ───
import re
st.header("🎼 Genre Breakdown")
# split multiple genres per track into a flat list
genre_series = df['Genres'].dropna().astype(str)
all_genres = genre_series.apply(lambda s: [g.strip() for g in re.split(r"[;,]", s) if g.strip()]).explode()
genre_counts = all_genres.value_counts().rename_axis('Genre').reset_index(name='Count')
# display top genres
top_n = st.slider("Number of top genres to display", min_value=5, max_value=20, value=10)
top_genres = genre_counts.head(top_n)
st.subheader(f"Top {top_n} Genres")
st.dataframe(top_genres)

# bar chart
fig_genre, ax_genre = plt.subplots()
ax_genre.barh(top_genres['Genre'][::-1], top_genres['Count'][::-1])
ax_genre.set_xlabel('Count')
ax_genre.set_title('Top Genres in Your Playlists')
st.pyplot(fig_genre)

# ─── Playlist Audio Feature Clustering ───
st.header("🧭 Playlist Landscape Map (PCA)")

from sklearn.decomposition import PCA

# Only run if there's more than 1 playlist
if len(avgs) > 1:
    features_for_pca = selected_feats.copy()
    if "Tempo" in features_for_pca:
        features_for_pca.remove("Tempo")  # optional to drop tempo if it skews scale

    feature_matrix = avgs[features_for_pca].values

    # PCA to 2 components
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(feature_matrix)

    # Plot
    fig_pca, ax_pca = plt.subplots()
    ax_pca.scatter(pca_coords[:, 0], pca_coords[:, 1], color='mediumslateblue')

    for i, name in enumerate(avgs.index):
        ax_pca.text(pca_coords[i, 0], pca_coords[i, 1], name, fontsize=9, ha='center', va='center')

    ax_pca.set_title("🎯 Playlist Clustering via Audio Features")
    ax_pca.set_xlabel("PCA Component 1")
    ax_pca.set_ylabel("PCA Component 2")

    st.pyplot(fig_pca)
else:
    st.info("At least 2 playlists are needed to compute PCA positioning.")

# ─── Most Distinctive Tracks by Feature ───
st.header("🧬 Unique Tracks: Vocals, Volume & Vibes")

# Smart detection of column names
track_col = next((col for col in df.columns if col.lower().strip() in ['track name', 'track']), None)
artist_col = next((col for col in df.columns if col.lower().strip() == 'artist'), None)

# Proceed only if required columns are present
if track_col and artist_col:
    outliers_df = df[['Playlist', track_col, artist_col, 'Speechiness', 'Instrumentalness', 'Loudness', 'Energy']].dropna()

    # 🎙️ Speechiest Tracks
    st.subheader("🎙️ Most Speech-Driven Tracks")
    speechy = outliers_df.sort_values(by='Speechiness', ascending=False).head(5)
    st.dataframe(speechy[[track_col, artist_col, 'Speechiness']])

    # 🎻 Most Instrumental
    st.subheader("🎻 Most Instrumental Tracks")
    instr = outliers_df.sort_values(by='Instrumentalness', ascending=False).head(5)
    st.dataframe(instr[[track_col, artist_col, 'Instrumentalness']])

    # 🔊 Loudest
    st.subheader("🔊 Loudest Tracks")
    loudest = outliers_df.sort_values(by='Loudness', ascending=False).head(5)
    st.dataframe(loudest[[track_col, artist_col, 'Loudness']])

    # 🔈 Quietest
    st.subheader("🔈 Quietest Tracks")
    quietest = outliers_df.sort_values(by='Loudness', ascending=True).head(5)
    st.dataframe(quietest[[track_col, artist_col, 'Loudness']])

    # ⚡ Highest Energy
    st.subheader("⚡ Highest Energy Tracks")
    high_energy = outliers_df.sort_values(by='Energy', ascending=False).head(5)
    st.dataframe(high_energy[[track_col, artist_col, 'Energy']])

else:
    st.warning("Missing 'Track' or 'Artist' column in your data.")

# ─── 📈 Multi-Playlist Feature Timeline Explorer ───
st.header("📈 Feature Timeline Explorer")

feature_options = ['Energy', 'Valence', 'Danceability', 'Acousticness',
                   'Instrumentalness', 'Speechiness', 'Loudness']

if 'Playlist' in df.columns and 'AddedAt' in df.columns and 'ReleaseDate' in df.columns:

    playlist_choices = st.multiselect("Select Playlist(s)", df['Playlist'].unique(), default=df['Playlist'].unique()[:2])
    feature_choice = st.selectbox("Select Audio Feature", feature_options, key="timeline_feature")
    time_basis = st.radio("Time axis based on:", ["Date Added", "Release Date"], horizontal=True)
    show_rolling = st.checkbox("Show 5-point rolling average", value=True)

    time_col = 'AddedAt' if time_basis == "Date Added" else 'ReleaseDate'
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    fig, ax = plt.subplots(figsize=(10, 5))

    color_cycle = plt.cm.get_cmap("tab10", len(playlist_choices))

    for i, playlist in enumerate(playlist_choices):
        subset = df[df['Playlist'] == playlist].copy()
        subset = subset.dropna(subset=[time_col, feature_choice])
        subset = subset.sort_values(by=time_col)

        if not subset.empty:
            ax.plot(subset[time_col], subset[feature_choice],
                    marker='o', linestyle='-', alpha=0.5,
                    label=f"{playlist}", color=color_cycle(i))

            if show_rolling and len(subset) >= 5:
                subset['RollingAvg'] = subset[feature_choice].rolling(window=5).mean()
                ax.plot(subset[time_col], subset['RollingAvg'],
                        linestyle='--', linewidth=2, color=color_cycle(i))

            # Highlight peaks
            peak = subset.loc[subset[feature_choice].idxmax()]
            low = subset.loc[subset[feature_choice].idxmin()]
            ax.axhline(peak[feature_choice], color=color_cycle(i), linestyle=':', alpha=0.4)
            ax.axhline(low[feature_choice], color=color_cycle(i), linestyle=':', alpha=0.2)

    ax.set_title(f"{feature_choice} Over Time", fontsize=14)
    ax.set_ylabel(feature_choice)
    ax.set_xlabel(time_col)
    ax.grid(True)
    ax.legend(loc='best')
    st.pyplot(fig)
else:
    st.warning("Missing Playlist, AddedAt, or ReleaseDate columns.")

# ─── 🥧 Genre Spread Pie Chart (Single or All Playlists) ───
st.header("🥧 Genre Spread")

if 'Genres' in df.columns and 'Playlist' in df.columns:

    all_playlists = df['Playlist'].unique().tolist()
    playlist_options = ["All Playlists Combined"] + all_playlists
    selected_playlist = st.selectbox("Select Playlist for Genre Breakdown", playlist_options)

    if selected_playlist == "All Playlists Combined":
        genre_df = df.copy()
    else:
        genre_df = df[df['Playlist'] == selected_playlist].copy()

    # Ensure clean genres
    genre_df['Genres'] = genre_df['Genres'].dropna().astype(str)
    genre_series = genre_df['Genres'].str.split(',').explode().str.strip()

    # Get top genres
    top_genres = genre_series.value_counts().head(10)

    if not top_genres.empty:
        fig_genre, ax_genre = plt.subplots()
        ax_genre.pie(top_genres.values,
                     labels=top_genres.index,
                     autopct='%1.1f%%',
                     startangle=140,
                     counterclock=False)
        chart_title = "Top Genres Across All Playlists" if selected_playlist == "All Playlists Combined" else f"Top Genres in '{selected_playlist}'"
        ax_genre.set_title(chart_title)
        st.pyplot(fig_genre)
    else:
        st.info("No genre data found in this playlist.")

else:
    st.warning("Missing 'Genres' or 'Playlist' column in your data.")



# ─── Track Popularity & "Hidden Gems" ───
st.header("⭐ Track Popularity & Hidden Gems")
if 'Popularity' in df.columns:
    # Popularity distribution
    fig_pop, ax_pop = plt.subplots()
    df['Popularity'].hist(ax=ax_pop, bins=20)
    ax_pop.set_xlabel('Popularity')
    ax_pop.set_ylabel('Track Count')
    ax_pop.set_title('Popularity Distribution')
    st.pyplot(fig_pop)

    # Hidden gems criteria
    pop_thresh = st.slider("Max popularity for hidden gems", 0, 100, 30)
    energy_thresh = st.slider("Min energy for hidden gems", 0.0, 1.0, 0.7, 0.05)
    gems = df[(df['Popularity'] <= pop_thresh) & (df.get('Energy', 0) >= energy_thresh)]
    st.subheader(f"Hidden Gems (Pop ≤ {pop_thresh} & Energy ≥ {energy_thresh})")
    st.dataframe(
        gems[['Playlist','Track','Artist','Popularity','Energy']]
        .sort_values(['Energy','Popularity'], ascending=[False, True])
        .reset_index(drop=True)
        .head(20)
    )
else:
    st.warning("No 'Popularity' column found in your data for popularity analysis.")


# ─── Tempo Heatmap by Playlist ───
st.header("🎚️ Tempo Heatmap")

if 'Tempo' in df.columns and 'Playlist' in df.columns:
    tempo_avg = df.groupby('Playlist')['Tempo'].mean().sort_values()
    fig_tempo, ax_tempo = plt.subplots(figsize=(8, 5))
    sns.heatmap(tempo_avg.to_frame().T, cmap="magma", annot=True, fmt=".0f", cbar=True, linewidths=1, ax=ax_tempo)
    ax_tempo.set_title("Average Tempo by Playlist")
    ax_tempo.set_ylabel("")
    st.pyplot(fig_tempo)
else:
    st.warning("Tempo or Playlist column not found in the data.")


# Key & Tempo Trajectory
st.header("🎹 Key & Tempo Trajectory")
if 'Tempo' in df:
    dfo=df.sort_values('AddedAt'); dfo['TempoRoll']=dfo.groupby('Playlist')['Tempo'].transform(lambda x:x.rolling(10,min_periods=1).mean())
    figt, axt=plt.subplots(figsize=(10,4))
    for p in selected: axt.plot(dfo[dfo['Playlist']==p]['AddedAt'],dfo[dfo['Playlist']==p]['TempoRoll'],label=p)
    st.pyplot(figt)
if 'Key' in df:
    df['ReleaseYear']=df['ReleaseDate'].dt.year
    km=df.groupby(['ReleaseYear',df['Key'].map({0:'C',1:'C♯/D♭',2:'D',3:'D♯/E♭',4:'E',5:'F',6:'F♯/G♭',7:'G',8:'G♯/A♭',9:'A',10:'A♯/B♭',11:'B'})]).size().unstack(fill_value=0)
    st.subheader("Key Distribution by Release Year"); st.dataframe(km)

# Major vs Minor by Release Year
if 'Mode' in df and 'ReleaseDate' in df:
    df['ReleaseYear']=df['ReleaseDate'].dt.year
    mc=df.groupby(['ReleaseYear','Mode']).size().unstack(fill_value=0)
    st.subheader("Major vs. Minor by Release Year")
    figm, axm=plt.subplots(); mc.plot(kind='bar',stacked=True,ax=axm); st.pyplot(figm)



# ─── Listening Session Simulator ───
st.header("🎧 Listening Session Simulator")
# Define time-of-day slots with energy and valence ranges
slots = {
    "Morning (6–9 AM)": {"energy": (0.0, 0.5), "valence": (0.5, 1.0)},
    "Afternoon (9 AM–5 PM)": {"energy": (0.3, 0.7), "valence": (0.4, 0.8)},
    "Evening (5–9 PM)": {"energy": (0.5, 1.0), "valence": (0.3, 0.7)},
    "Night (9 PM–6 AM)": {"energy": (0.0, 0.4), "valence": (0.2, 0.6)}
}
# User-configurable number of tracks per slot
num_tracks = st.slider("How many tracks per slot", min_value=3, max_value=10, value=5)
# Sample tracks for each slot
for slot, params in slots.items():
    candidates = df[
        df['Energy'].between(*params['energy']) &
        df['Valence'].between(*params['valence'])
    ]
    st.subheader(f"{slot} — {len(candidates)} candidates")
    if not candidates.empty:
        sample = candidates[['Track','Artist','Energy','Valence']]
        st.table(
            sample.sample(
                n=min(num_tracks, len(sample)),
                random_state=42
            )
        )

# ─── Cross-Playlist Radar Comparison ───
st.header("🕸️ Cross-Playlist Radar Comparison")

# Select up to 4 playlists for radar
radar_choices = st.multiselect(
    "Pick up to 4 playlists to compare", df['Playlist'].unique().tolist(), default=selected[:2]
)

if radar_choices:
    # Remove 'Tempo' from selected features if it's present
    metrics = [m for m in selected_feats if m.lower() != 'tempo']

    avg_vals = df[df['Playlist'].isin(radar_choices)].groupby('Playlist')[metrics].mean()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(subplot_kw={"polar": True}, figsize=(6,6))

    for pname in radar_choices[:4]:
        vals = avg_vals.loc[pname].tolist()
        vals += vals[:1]
        ax_radar.plot(angles, vals, label=pname)
        ax_radar.fill(angles, vals, alpha=0.1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics)
    ax_radar.set_title("Audio Feature Radar Comparison")
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig_radar)


# ─── Genre Evolution Timeline ───
st.header("🎼 Genre Evolution Over Time")

if 'Genres' in df.columns and 'AddedAt' in df.columns:
    genre_timeline = df.dropna(subset=['Genres', 'AddedAt']).copy()
    genre_timeline['AddedMonth'] = genre_timeline['AddedAt'].dt.to_period("M").dt.to_timestamp()

    genre_counts = (
        genre_timeline.groupby(['AddedMonth', 'Genres'])
        .size()
        .reset_index(name='Count')
    )

    pivot = genre_counts.pivot(index='AddedMonth', columns='Genres', values='Count').fillna(0)

    # Limit to top 5 genres for readability
    top_genres = pivot.sum().sort_values(ascending=False).head(5).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot[top_genres].plot.area(ax=ax)
    ax.set_ylabel("# of Tracks")
    ax.set_title("Top 5 Genres Added Over Time")
    ax.legend(title="Genre")
    st.pyplot(fig)

else:
    st.info("No genre or added-at date info available to show genre evolution.")

# ─── Emotional Journey Curve ───
st.header("📈 Emotional Journey Over Time")

if 'Energy' in df.columns and 'Valence' in df.columns and 'AddedAt' in df.columns:
    mood_df = df.dropna(subset=['Energy', 'Valence', 'AddedAt']).copy()
    mood_df['AddedDate'] = mood_df['AddedAt'].dt.to_period("M").dt.to_timestamp()

    mood_summary = (
        mood_df.groupby('AddedDate')[['Energy', 'Valence']]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots()
    ax.plot(mood_summary['AddedDate'], mood_summary['Energy'], label='Energy', marker='o')
    ax.plot(mood_summary['AddedDate'], mood_summary['Valence'], label='Valence (Happiness)', marker='s')
    ax.set_title("Your Emotional Music Journey")
    ax.set_ylabel("Average Value (0–1)")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Not enough data to plot emotional journey.")

# ─── Mood Profile Clusters ───
st.header("🧠 Your Top 5 Mood Profiles")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if all(col in df.columns for col in ['Energy', 'Valence', 'Danceability']):
    mood_features = df[['Energy', 'Valence', 'Danceability']].dropna()
    mood_tracks = df.loc[mood_features.index][['Track', 'Artist', 'Playlist']].reset_index(drop=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(mood_features)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    mood_tracks['Mood Cluster'] = kmeans.fit_predict(scaled)

    # Assign rough mood labels (you can tweak these!)
    labels = {
        0: "💃 Club Bangers",
        1: "☁️ Chill Vibes",
        2: "🌧 Introspective",
        3: "🏋️ Hype Mode",
        4: "🎨 Artsy / Offbeat"
    }
    mood_tracks['Mood'] = mood_tracks['Mood Cluster'].map(labels)

    top_moods = (
        mood_tracks['Mood'].value_counts()
        .head(5)
        .reset_index(name='Track Count')
        .rename(columns={'index': 'Mood'})
    )

    st.dataframe(top_moods)

    st.subheader("🎧 Sample Track per Mood")
    for mood in top_moods['Mood']:
        track = mood_tracks[mood_tracks['Mood'] == mood].sample(1).iloc[0]
        st.markdown(f"**{mood}** → *{track['Track']}* by *{track['Artist']}*")
else:
    st.info("Not enough data to generate mood profiles.")

# ─── Audio Fingerprint Comparison ───
st.header("🆚 Audio Fingerprint Comparison")

compare_choices = st.multiselect(
    "Pick 2 playlists to compare",
    df['Playlist'].unique().tolist(),
    default=selected[:2]
)

if len(compare_choices) == 2:
    p1, p2 = compare_choices
    p1_avg = df[df['Playlist'] == p1][selected_feats].mean()
    p2_avg = df[df['Playlist'] == p2][selected_feats].mean()

    delta = (p1_avg - p2_avg).sort_values(key=lambda x: abs(x), ascending=False)
    biggest_diffs = delta.head(5).index.tolist()

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(biggest_diffs))
    ax.bar(x - width/2, p1_avg[biggest_diffs], width, label=p1)
    ax.bar(x + width/2, p2_avg[biggest_diffs], width, label=p2)
    ax.set_xticks(x)
    ax.set_xticklabels(biggest_diffs, rotation=45)
    ax.set_ylabel("Average Value")
    ax.set_title("Top 5 Audio Feature Differences")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please select exactly 2 playlists to compare.")

# ─── Artist Depth Meter ───
st.header("🔍 Artist Depth Meter")

if 'Artist' in df.columns and 'Popularity' in df.columns:
    # Expand multi-artist tracks
    rows = []
    for _, row in df.dropna(subset=['Artist']).iterrows():
        artists = [a.strip() for a in str(row['Artist']).split(',')]
        for artist in artists:
            rows.append({
                'Artist': artist,
                'Track': row['Track'],
                'Popularity': row.get('Popularity', np.nan)
            })

    depth_df = pd.DataFrame(rows)
    artist_summary = (
        depth_df.groupby('Artist')
        .agg({'Track': 'count', 'Popularity': 'mean'})
        .rename(columns={'Track': 'Track Count', 'Popularity': 'Avg Popularity'})
        .sort_values('Track Count', ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(artist_summary.style.format({'Avg Popularity': '{:.1f}'}))

    fig, ax = plt.subplots()
    ax.scatter(
        artist_summary['Track Count'],
        artist_summary['Avg Popularity'],
        s=100
    )
    for _, row in artist_summary.iterrows():
        ax.text(row['Track Count']+0.1, row['Avg Popularity'], row['Artist'], fontsize=8)
    ax.set_xlabel("Track Count")
    ax.set_ylabel("Avg Popularity")
    ax.set_title("Your Listening Depth by Artist")
    st.pyplot(fig)
else:
    st.info("Artist or popularity data missing.")

# ─── Tempo vs Loudness Heatmap ───
st.header("📊 Tempo vs Loudness Heatmap")

if 'Tempo' in df.columns and 'Loudness' in df.columns:
    tempo_loud_df = df.dropna(subset=['Tempo', 'Loudness'])

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        tempo_loud_df['Tempo'],
        tempo_loud_df['Loudness'],
        alpha=0.5
    )
    ax.set_xlabel("Tempo (BPM)")
    ax.set_ylabel("Loudness (dB)")
    ax.set_title("How Loud and Fast is Your Music?")
    st.pyplot(fig)
else:
    st.info("Tempo and loudness data not available.")

# ─── Playlist Rarity Index ───
st.header("🧬 Playlist Rarity Index")

if 'Popularity' in df.columns:
    rarity = (
        df.dropna(subset=['Popularity'])
        .groupby('Playlist')['Popularity']
        .mean()
        .sort_values()
        .round(1)
    )

    fig, ax = plt.subplots()
    rarity.plot(kind='barh', ax=ax, color='purple')
    ax.set_xlabel("Average Track Popularity (0 = rare, 100 = mainstream)")
    ax.set_title("Which Playlists Are the Most Underground?")
    st.pyplot(fig)

    st.caption("Lower = rarer. Spotify popularity score is based on streams and trending status.")
else:
    st.info("Popularity data is missing. Can't compute rarity.")

# ─── Time Travel Playlist by Decade ───
st.header("📻 Time Travel by Decade")

if 'ReleaseDate' in df.columns:
    decades_df = df.dropna(subset=['ReleaseDate']).copy()
    decades_df['Year'] = pd.to_datetime(decades_df['ReleaseDate'], errors='coerce').dt.year
    decades_df['Decade'] = (decades_df['Year'] // 10 * 10).astype('Int64')

    decade_counts = (
        decades_df['Decade']
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots()
    decade_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Track Count")
    ax.set_title("Your Tracks by Decade of Release")
    st.pyplot(fig)

    st.subheader("🎵 Top Track from Each Decade")
    for decade in decade_counts.index:
        track = decades_df[decades_df['Decade'] == decade].sample(1).iloc[0]
        st.markdown(f"**{decade}s** → *{track['Track']}* by *{track['Artist']}*")

else:
    st.info("Release dates not available for time travel.")

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from matplotlib import font_manager as fm
import os

# ─── Festival Poster Generator ───
st.header("🎪 Downloadable Festival Poster")

# 🎨 Background and Font Options
bg_options = {
    "Space Rave": ("space_rave.jpg", "rave_font.ttf"),
    "Desert Dusk": ("desert_dusk.jpg", "desert_font.ttf"),
    "Glitch City": ("glitch_city.jpg", "glitch_font.ttf")
}

selected_theme = st.selectbox("Choose your poster style:", list(bg_options.keys()))
bg_path, font_path = bg_options[selected_theme]

# Load the font
if os.path.exists(font_path):
    custom_font = fm.FontProperties(fname=font_path)
else:
    custom_font = fm.FontProperties(family='monospace')

# ─── Poster Logic ───
if 'Artist' in df.columns:
    from collections import Counter

    # Get top 40 artists
    all_artists = []
    for a in df['Artist'].dropna():
        all_artists.extend([x.strip() for x in str(a).split(',')])
    artist_counts = Counter(all_artists)
    top_40 = artist_counts.most_common(40)
    top_artists = [a for a, _ in top_40]

    midpoint = len(top_artists) // 3
    day_1 = top_artists[:midpoint]
    day_2 = top_artists[midpoint:]

    # ─── Build Poster ───
    def build_poster(day1, day2):
        text_color = 'white'
        if selected_theme in ['Desert Dusk', 'Glitch City']:
            text_color = 'yellow'

        fig, ax = plt.subplots(figsize=(10, 16))
        ax.axis('off')

        # Background image
        if os.path.exists(bg_path):
            bg_img = Image.open(bg_path)
            ax.imshow(bg_img, extent=[0, 1, 0, 1], aspect='auto', zorder=0)
        else:
            ax.set_facecolor("#121212")
            fig.patch.set_facecolor("#121212")

        # Add translucent background overlay for better readability
        ax.add_patch(plt.Rectangle((0, 0.42), 1, 0.52, color='black', alpha=0.3, zorder=1))

        # Draw day section
        def draw_day(title, artists, y_start):
            ax.text(0.5, y_start, title, fontsize=24, fontweight='bold', ha='center',
                    color=text_color, zorder=2, fontproperties=custom_font)
            ax.text(0.5, y_start - 0.05, artists[0], fontsize=34, fontweight='bold', ha='center',
                    color='gold', zorder=2, fontproperties=custom_font)
            ax.text(0.5, y_start - 0.10, ' • '.join(artists[1:4]), fontsize=18, ha='center',
                    color=text_color, zorder=2, fontproperties=custom_font)
            ax.text(0.5, y_start - 0.15, ' • '.join(artists[4:8]), fontsize=14, ha='center',
                    color='lightgray', zorder=2, fontproperties=custom_font)

            # Handle overflow nicely
            long_artists = artists[8:]
            max_per_line = 3 if selected_theme == "Glitch City" else 3
            lines = [' • '.join(long_artists[i:i+max_per_line]) for i in range(0, len(long_artists), max_per_line)]
            for idx, line in enumerate(lines):
                ax.text(0.5, y_start - 0.22 - idx * 0.038, line, fontsize=12, ha='center',
                        color=text_color, zorder=2, fontproperties=custom_font)

        # Festival title with shadow
        for dx, dy in [(-0.002, -0.002), (0.002, -0.002), (-0.002, 0.002), (0.002, 0.002)]:
            ax.text(0.5 + dx, 0.96 + dy, "SONICMIRROR FESTIVAL", fontsize=28, fontweight='bold',
                    ha='center', color='black', zorder=1, fontproperties=custom_font)
        ax.text(0.5, 0.96, "SONICMIRROR FESTIVAL", fontsize=28, fontweight='bold',
                ha='center', color=text_color, zorder=2, fontproperties=custom_font)

        draw_day("DAY 1", day1, 0.88)
        draw_day("DAY 2", day2, 0.48)

        # Output buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf

    # Generate and show
    poster_buf = build_poster(day_1, day_2)
    st.image(poster_buf, caption="Your Festival Lineup", use_container_width=True)
    st.download_button("📥 Download Poster", poster_buf, file_name="sonicmirror_festival.png", mime="image/png")

else:
    st.info("Artist data missing — can't build your lineup.")


# --- Vibe Moodboard from Actual Audio Features ---
st.header("🎭 Vibe Moodboard")

# Make sure required features exist
required_cols = ["Valence", "Energy", "Danceability", "Acousticness", "Speechiness", "Instrumentalness"]
if not all(col in df.columns for col in required_cols):
    st.warning("Not enough audio features available to build the Vibe Moodboard.")
else:
    def classify_mood(row):
        valence = row["Valence"]
        energy = row["Energy"]
        dance = row["Danceability"]
        acoustic = row["Acousticness"]
        speech = row["Speechiness"]
        instr = row["Instrumentalness"]

        if valence < 0.3 and energy < 0.4:
            return "😢 Feels Trip"
        elif energy > 0.7 and valence > 0.6 and dance > 0.7:
            return "🔥 Hype Mode"
        elif valence > 0.5 and dance > 0.6:
            return "🎉 Party Time"
        elif valence < 0.4 and dance < 0.4 and speech < 0.5:
            return "🧠 Introspective"
        elif energy < 0.4 and acoustic > 0.6:
            return "😎 Chill Zone"
        elif instr > 0.6 and speech < 0.3 and energy < 0.6:
            return "🌌 Dreamwave"
        else:
            return "🧠 Introspective"  # fallback

    # Apply mood classifier to each song
    df["Mood"] = df.apply(classify_mood, axis=1)

    # Tally the moods
    mood_counts = df["Mood"].value_counts().to_dict()
    total = sum(mood_counts.values())

    # Sort by frequency
    sorted_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)

    # Mood styles
    mood_styles = {
        "🔥 Hype Mode": ("#ff5733", "🔥"),
        "😎 Chill Zone": ("#5dade2", "😎"),
        "😢 Feels Trip": ("#8e44ad", "😢"),
        "🎉 Party Time": ("#f39c12", "🎉"),
        "🧠 Introspective": ("#34495e", "🧠"),
        "🌌 Dreamwave": ("#1abc9c", "🌌")
    }

    # Mood summary
    if total > 0:
        dominant = sorted_moods[0][0]
        secondary = sorted_moods[1][0] if len(sorted_moods) > 1 else ""
        st.markdown(f"""
        <div style="font-size: 1.2em; margin-bottom: 1em;">
            🎶 This playlist leans <strong>{dominant[2:]}</strong> — mostly {dominant} with a touch of {secondary}
        </div>
        """, unsafe_allow_html=True)

        # Mood chart bars
        for mood, count in sorted_moods:
            percent = int((count / total) * 100)
            color, emoji = mood_styles.get(mood, ("#888", "🎧"))
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.3em 0;">
                <div style="width: 3em; font-size: 1.5em;">{emoji}</div>
                <div style="flex: 1; background: #eee; border-radius: 10px; overflow: hidden;">
                    <div style="width: {percent}%; background: {color}; padding: 0.3em; color: white; font-weight: bold; text-align: right;">
                        {percent}%
                    </div>
                </div>
                <div style="margin-left: 1em; font-weight: 600;">{mood[2:]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No valid mood data found in the playlist.")

# ─── 📆 Track Duration Over Time ───
st.header("📆 Track Duration Over Time")

# Check required columns
if 'ReleaseDate' in df.columns and 'Duration (ms)' in df.columns:

    df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
    df['ReleaseYear'] = df['ReleaseDate'].dt.year
    df = df[df['Duration (ms)'] > 0]
    df['Duration (min)'] = df['Duration (ms)'] / 60000

    filtered = df.dropna(subset=['ReleaseYear', 'Duration (min)'])

    if not filtered.empty:
        # Identify longest and shortest tracks
        track_col = next((col for col in df.columns if col.lower() in ['track name', 'track']), 'Unknown')
        artist_col = next((col for col in df.columns if col.lower() == 'artist'), 'Unknown')

        longest = filtered.loc[filtered['Duration (min)'].idxmax()]
        shortest = filtered.loc[filtered['Duration (min)'].idxmin()]

        # Display insights
        st.markdown(f"**📏 Longest Track:** {longest[track_col]} by {longest[artist_col]} — {longest['Duration (min)']:.2f} min ({int(longest['ReleaseYear'])})")
        st.markdown(f"**🐜 Shortest Track:** {shortest[track_col]} by {shortest[artist_col]} — {shortest['Duration (min)']:.2f} min ({int(shortest['ReleaseYear'])})")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=filtered,
            x='ReleaseYear',
            y='Duration (min)',
            hue='Duration (min)',
            palette='viridis',
            legend=False,
            alpha=0.7,
            ax=ax
        )

        ax.set_title("Track Duration vs. Release Year", fontsize=16, weight='bold')
        ax.set_xlabel("Release Year")
        ax.set_ylabel("Duration (minutes)")
        ax.grid(True, linestyle='--', alpha=0.3)

        st.pyplot(fig)
    else:
        st.info("No valid release year or duration data available.")
else:
    st.warning("This feature requires both 'ReleaseDate' and 'Duration (ms)' columns.")

# ─── 🤝 Top Collaborations Detector ───
st.header("🤝 Top Collaborations")

if 'Artist' in df.columns:
    from collections import Counter
    import itertools

    # Split artists and explode
    artist_lists = df['Artist'].dropna().apply(lambda x: [a.strip() for a in str(x).split(',')])
    df['ArtistList'] = artist_lists

    # Count all pairs
    all_pairs = []
    for artist_group in artist_lists:
        if len(artist_group) > 1:
            all_pairs.extend(itertools.combinations(sorted(set(artist_group)), 2))

    pair_counts = Counter(all_pairs)
    top_pairs = pair_counts.most_common(10)

    st.subheader("👯 Most Frequent Collaborating Artist Pairs")
    if top_pairs:
        for (a1, a2), count in top_pairs:
            st.markdown(f"- **{a1}** + **{a2}** → {count} tracks")
    else:
        st.info("No recurring artist pairs found.")

    # Songs with 3+ collaborators
    multi_collabs = df[df['ArtistList'].apply(lambda x: len(x) >= 3)]
    st.subheader("🎤 Songs with 3 or More Collaborators")
    if not multi_collabs.empty:
        display_cols = [col for col in ['Track Name', 'Artist', 'Playlist'] if col in df.columns]
        st.dataframe(multi_collabs[display_cols])
    else:
        st.info("No songs found with 3 or more listed artists.")

    # Artists with most collabs
    all_artists_flat = list(itertools.chain.from_iterable(artist_lists))
    artist_count = Counter(all_artists_flat)
    collab_counts = {k: v for k, v in artist_count.items() if v > 1}
    top_collab_artists = Counter(collab_counts).most_common(10)

    st.subheader("🏆 Artists with Most Collaborations")
    if top_collab_artists:
        for artist, count in top_collab_artists:
            st.markdown(f"- **{artist}** → {count} total appearances")
    else:
        st.info("No collaborating artists found more than once.")
else:
    st.warning("Missing 'Artist' column in your data.")

# ─── 🧠 Expanded Playlist Personality Engine ───
st.header("🧠 Playlist Personality Profile")

features = ['Energy', 'Valence', 'Danceability', 'Acousticness', 'Speechiness', 'Loudness']

if 'Playlist' in df.columns and all(f in df.columns for f in features):
    selected_playlist = st.selectbox("Select a playlist to analyze", df['Playlist'].unique(), key="personality")

    avg = df[df['Playlist'] == selected_playlist][features].mean()

    # Define more granular rules
    personality = "🌀 The Eclectic – A blend of moods, unpredictable but fun."

    if avg['Valence'] > 0.7 and avg['Energy'] > 0.7 and avg['Danceability'] > 0.7:
        personality = "🎉 The Party Starter – Pure joy, volume up, body moving!"
    elif avg['Valence'] > 0.7 and avg['Acousticness'] > 0.6:
        personality = "🌅 The Sunset Rider – Warm, nostalgic, and low-key romantic."
    elif avg['Valence'] < 0.3 and avg['Acousticness'] > 0.5:
        personality = "💔 The Heartbreaker – Sad songs for soft souls."
    elif avg['Energy'] > 0.8 and avg['Speechiness'] > 0.5:
        personality = "🎤 The Mic Dropper – Confident, expressive, and lyrically fire."
    elif avg['Acousticness'] > 0.75 and avg['Loudness'] < -10:
        personality = "🧘 The Minimalist – Quiet, introspective, unplugged."
    elif avg['Danceability'] > 0.7 and avg['Valence'] < 0.4:
        personality = "🕶️ The Cool Customer – Groovy but emotionally reserved."
    elif avg['Loudness'] > -5 and avg['Energy'] > 0.6 and avg['Speechiness'] < 0.3:
        personality = "⚡ The Wild One – Loud, fast, and full throttle."
    elif avg['Energy'] < 0.4 and avg['Valence'] < 0.4 and avg['Acousticness'] > 0.6:
        personality = "🌧 The Moody Introvert – Introspective and emotionally deep."
    elif avg['Energy'] < 0.3 and avg['Valence'] > 0.7:
        personality = "🦋 The Gentle Optimist – Soft, bright, and hopeful."
    elif avg['Danceability'] > 0.5 and avg['Speechiness'] > 0.6:
        personality = "🎙️ The Open Mic Regular – Wordy, clever, and expressive."

    # Show result
    st.subheader(f"Personality of '{selected_playlist}'")
    st.markdown(f"**{personality}**")
    st.write("📊 Feature Profile:")
    st.dataframe(avg.round(3).to_frame("Average Value"))
else:
    st.warning("Missing required audio features.")

# ─── 🧬 MBTI Playlist Personality Decoder ───
st.header("🧬 MBTI Personality Match")

mbti_features = ['Energy', 'Valence', 'Danceability', 'Acousticness',
                 'Instrumentalness', 'Speechiness', 'Tempo', 'LagDays']

if all(f in df.columns for f in mbti_features):
    mbti_playlist = st.selectbox("Select a playlist for MBTI profiling", df['Playlist'].unique(), key="mbti_select")

    avg = df[df['Playlist'] == mbti_playlist][mbti_features].mean()

 # ─ Visual Profile (Exclude LagDays and Tempo from chart) ─
display_features = [f for f in mbti_features if f not in ['LagDays', 'Tempo']]
norm_avg_display = avg[display_features]

# Normalize these features only (0–1 scaling)
norm_avg_display = (norm_avg_display - norm_avg_display.min()) / (norm_avg_display.max() - norm_avg_display.min())

st.subheader("📊 Feature Profile (Normalized, No LagDays or Tempo)")
fig_bar, ax = plt.subplots()
ax.barh(norm_avg_display.index[::-1], norm_avg_display.values[::-1], color='slateblue')
ax.set_xlim(0, 1)
ax.set_xlabel("Normalized Value (0–1)")
ax.set_title("Audio Feature Profile (Excludes Tempo & LagDays)")
st.pyplot(fig_bar)



    # ─ Custom Threshold Sliders ─
with st.expander("🎛 Fine-tune MBTI Thresholds"):
    energy_thresh = st.slider("Extroversion (Energy)", 0.3, 0.9, 0.55)
    speech_thresh = st.slider("Extroversion (Speechiness)", 0.2, 0.7, 0.4)
    instr_thresh = st.slider("Intuition (Instrumentalness)", 0.1, 0.8, 0.3)
    acoustic_thresh = st.slider("Intuition (Acousticness)", 0.3, 0.9, 0.5)
    valence_thresh = st.slider("Feeling (Valence)", 0.3, 0.7, 0.5)
    tempo_std_thresh = st.slider("Perceiving (Tempo Std Dev)", 5, 40, 15)
    lag_std_thresh = st.slider("Perceiving (Lag Std Dev)", 30, 150, 100)

    # ─ MBTI Logic ─
    ie = "E" if avg['Energy'] > energy_thresh or avg['Speechiness'] > speech_thresh else "I"
    ns = "N" if avg['Instrumentalness'] > instr_thresh or avg['Acousticness'] > acoustic_thresh else "S"
    tf = "F" if avg['Valence'] > valence_thresh else "T"
    tempo_std = df[df['Playlist'] == mbti_playlist]['Tempo'].std()
    lag_std = df[df['Playlist'] == mbti_playlist]['LagDays'].std()
    jp = "P" if tempo_std > tempo_std_thresh or lag_std > lag_std_thresh else "J"

    mbti = ie + ns + tf + jp

    # ─ Expanded MBTI Profiles ─
    mbti_profiles = {
        "ENFP": "🎉 **The Sonic Adventurer** – Bursting with color, contradictions, and joy. Your playlist is a wild road trip with zero regrets.",
        "INFP": "🌌 **The Dream Weaver** – Gentle, introspective, and emotionally layered. Each song is a star in your inner universe.",
        "INTJ": "🧠 **The Sonic Architect** – Structured and intentional. Every song feels chosen by design, not impulse.",
        "ENTP": "⚡ **The Idea Storm** – Genre-fluid and chaos-curious. You collect sound like a dragon hoards gold: brilliantly, and with flair.",
        "ESFJ": "❤️ **The Harmonizer** – Warm and inviting. You build playlists like dinner parties – for shared joy, not just yourself.",
        "ISFJ": "🌿 **The Nostalgic Soul** – Gentle acoustics and emotional safety. You cherish sound memories like handwritten letters.",
        "ESTP": "🔥 **The Night Burner** – Loud, kinetic, unstoppable. Your tracks feel like the pre-drop before a riot of glitter.",
        "ISTP": "😎 **The Chill Tinkerer** – Precise, sleek, and unexpected. Minimal but deeply effective.",
        "INFJ": "🪐 **The Visionary Poet** – Obscure and layered. You find spiritual symmetry in ambient tones and glitchy metaphors.",
        "ENFJ": "🌞 **The Sound Shepherd** – Bold but loving. These tracks guide the vibe like an emotional compass.",
        "ISTJ": "📘 **The Archivist** – Structured, timeless, practical. You revere consistency in rhythm and mood.",
        "ESTJ": "📣 **The Commander** – Direct and high-intensity. Songs that wake people up. Not every track has lyrics, but every one has orders.",
        "INTP": "🔍 **The Analyst** – Curious and eclectic. You follow strange melodies like breadcrumb trails through a sonic forest.",
        "ENTJ": "🚀 **The Trailblazer** – Strategic and sharp-edged. Even your chill songs march with intent.",
    }

    # ─ MBTI Decoder ─
st.header("🧬 MBTI Personality Match")

mbti_features = ['Energy', 'Valence', 'Danceability', 'Acousticness',
                 'Instrumentalness', 'Speechiness', 'Tempo', 'LagDays']

if all(f in df.columns for f in mbti_features):
    mbti_playlist = st.selectbox("Select a playlist for MBTI profiling", df['Playlist'].unique(), key="mbti_select_unique")

    avg = df[df['Playlist'] == mbti_playlist][mbti_features].mean()

    # ... your bar chart, sliders, logic, and MBTI assignment ...

    # ─ Final Output ─
    st.subheader(f"🧬 Your Playlist MBTI Type: `{mbti}`")
    st.markdown(mbti_profiles.get(mbti, f"🌀 **The Enigma ({mbti})** – Unclassifiable, genre-resistant, and probably brilliant."))

    with st.expander("📖 Why This Type?"):
        st.markdown(f"""
        - **Introvert (I) vs Extrovert (E)** → Energy: `{avg['Energy']:.2f}`, Speechiness: `{avg['Speechiness']:.2f}`
        - **Intuitive (N) vs Sensing (S)** → Instrumentalness: `{avg['Instrumentalness']:.2f}`, Acousticness: `{avg['Acousticness']:.2f}`
        - **Thinking (T) vs Feeling (F)** → Valence: `{avg['Valence']:.2f}`
        - **Judging (J) vs Perceiving (P)** → Tempo Std Dev: `{tempo_std:.1f}`, Lag Std Dev: `{lag_std:.1f}`
        """)
        st.caption("You can adjust thresholds above to reflect different interpretations of your musical personality.")

else:
    st.warning("Not enough audio feature data to compute MBTI type.")

# ─── 🧠 Cognitive Bias Analyzer ───
st.header("🧠 Cognitive Bias Analyzer")

bias_messages = []

# ─ Recency Bias ─
if 'ReleaseDate' in df.columns:
    recent_threshold = pd.Timestamp.now() - pd.DateOffset(years=1)
    recent_pct = (df['ReleaseDate'] > recent_threshold).mean()
    if recent_pct > 0.5:
        bias_messages.append(f"🆕 **Recency Bias** – {int(recent_pct*100)}% of your tracks were released in the last year. You're chasing the new.")

# ─ Nostalgia Bias ─
if 'ReleaseDate' in df.columns:
    decades = df['ReleaseDate'].dt.year.dropna().dropna().floordiv(10) * 10
    if not decades.empty:
        top_decade = decades.mode()[0]
        top_decade_pct = (decades == top_decade).mean()
        if top_decade_pct > 0.4:
            bias_messages.append(f"📼 **Nostalgia Bias** – {int(top_decade_pct*100)}% of your tracks are from the {top_decade}s. Stuck in a beautiful time loop?")

# ─ Confirmation Bias ─
if 'Genres' in df.columns:
    genre_series = df['Genres'].dropna().astype(str).str.split(',').explode().str.strip()
    top_genre_pct = genre_series.value_counts(normalize=True).iloc[0] if not genre_series.empty else 0
    if top_genre_pct > 0.4:
        bias_messages.append(f"🧠 **Confirmation Bias** – {int(top_genre_pct*100)}% of your tracks share a single dominant genre. Comfortable much?")

# ─ Novelty Aversion ─
if 'Artist' in df.columns:
    artist_counts = df['Artist'].dropna().astype(str).value_counts()
    new_artists_pct = (artist_counts == 1).mean()
    if new_artists_pct < 0.4:
        bias_messages.append(f"😌 **Novelty Aversion** – Only {int(new_artists_pct*100)}% of artists appear once. You like who you like.")

# ─ Emotional Looping ─
if all(c in df.columns for c in ['Valence', 'Energy']):
    avg_val = df['Valence'].mean()
    avg_energy = df['Energy'].mean()
    if avg_val < 0.35 and avg_energy < 0.5:
        bias_messages.append(f"💔 **Emotional Looping** – Your vibe leans sad + chill. Do the tears taste like reverb?")
    elif avg_val > 0.65 and avg_energy > 0.6:
        bias_messages.append(f"🔥 **Emotional Looping** – You keep it hype and happy. Ride or die with the dopamine.")

# ─ Herd Bias ─
if 'Popularity' in df.columns:
    avg_popularity = df['Popularity'].dropna().mean()
    if avg_popularity > 70:
        bias_messages.append(f"👥 **Herd Bias** – Your average popularity is {avg_popularity:.0f}. You’re vibing with the masses.")

# ─ Results ─
if bias_messages:
    st.subheader("🪞 You Might Be Under the Influence Of:")
    for msg in bias_messages:
        st.markdown(f"- {msg}")
else:
    st.success("🎯 You're a musical free spirit — no obvious cognitive biases detected.")

# ─── 🧠 Cognitive Bias Radar Chart ───
st.header("🧠 Cognitive Bias Profile")

bias_features = ['Recency', 'Nostalgia', 'Confirmation', 'Novelty Aversion', 'Emotional Looping', 'Herd Following']
bias_scores = {b: 0 for b in bias_features}

bias_playlist = st.selectbox("Select playlist to analyze", df['Playlist'].unique(), key="bias_select")

df_bias = df[df['Playlist'] == bias_playlist].copy()

# Bias calculations
if 'ReleaseDate' in df_bias.columns:
    now = pd.Timestamp.now()
    one_year_ago = now - pd.DateOffset(years=1)
    recency = (df_bias['ReleaseDate'] > one_year_ago).mean()
    bias_scores['Recency'] = recency

    decades = df_bias['ReleaseDate'].dt.year.dropna().floordiv(10) * 10
    if not decades.empty:
        top_decade_pct = (decades == decades.mode()[0]).mean()
        bias_scores['Nostalgia'] = top_decade_pct

if 'Genres' in df_bias.columns:
    genre_series = df_bias['Genres'].dropna().astype(str).str.split(',').explode().str.strip()
    if not genre_series.empty:
        top_genre_pct = genre_series.value_counts(normalize=True).iloc[0]
        bias_scores['Confirmation'] = top_genre_pct

if 'Artist' in df_bias.columns:
    artist_counts = df_bias['Artist'].dropna().astype(str).value_counts()
    new_artist_pct = (artist_counts == 1).mean()
    bias_scores['Novelty Aversion'] = 1 - new_artist_pct  # inversion: low novelty = high aversion

if all(c in df_bias.columns for c in ['Valence', 'Energy']):
    val = df_bias['Valence'].mean()
    energy = df_bias['Energy'].mean()
    if val < 0.35 and energy < 0.5:
        bias_scores['Emotional Looping'] = 0.8
    elif val > 0.65 and energy > 0.6:
        bias_scores['Emotional Looping'] = 0.6
    else:
        bias_scores['Emotional Looping'] = 0.2

if 'Popularity' in df_bias.columns:
    pop = df_bias['Popularity'].dropna().mean()
    bias_scores['Herd Following'] = min(pop / 100, 1.0)

# ─ Radar Chart ─
angles = np.linspace(0, 2 * np.pi, len(bias_features), endpoint=False).tolist()
values = [bias_scores[b] for b in bias_features]
values += values[:1]
angles += angles[:1]

fig_bias, ax_bias = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax_bias.plot(angles, values, linewidth=2, color='crimson')
ax_bias.fill(angles, values, alpha=0.25, color='crimson')
ax_bias.set_xticks(angles[:-1])
ax_bias.set_xticklabels(bias_features)
ax_bias.set_title(f"Cognitive Bias Radar – {bias_playlist}")
ax_bias.set_ylim(0, 1)
st.pyplot(fig_bias)

# Optional table
if st.checkbox("Show bias scores as table"):
    st.dataframe(pd.DataFrame.from_dict(bias_scores, orient='index', columns=['Score']).round(2))


# ─── 🎛 Sonic Identity Dashboard ───
st.header("🎛 Sonic Identity Dashboard")

combo_playlist = st.selectbox("Select a playlist for deep soul analysis", df['Playlist'].unique(), key="combo_tabbed")
cdf = df[df['Playlist'] == combo_playlist].copy()

# ─ Common Feature Averages ─
val = cdf['Valence'].mean() if 'Valence' in cdf else 0.5
energy = cdf['Energy'].mean() if 'Energy' in cdf else 0.5
dance = cdf['Danceability'].mean() if 'Danceability' in cdf else 0.5
acoustic = cdf['Acousticness'].mean() if 'Acousticness' in cdf else 0.5
speech = cdf['Speechiness'].mean() if 'Speechiness' in cdf else 0.5
instr = cdf['Instrumentalness'].mean() if 'Instrumentalness' in cdf else 0.3
pop = cdf['Popularity'].mean() if 'Popularity' in cdf else 50
tempo_std = cdf['Tempo'].std() if 'Tempo' in cdf else 0
recent_pct = (cdf['ReleaseDate'] > pd.Timestamp.now() - pd.DateOffset(years=1)).mean() if 'ReleaseDate' in cdf else 0

# Year + Genre
if 'ReleaseDate' in cdf and not cdf['ReleaseDate'].isna().all():
    year = int(cdf['ReleaseDate'].dt.year.dropna().mode()[0])
    decade = (year // 10) * 10
else:
    year = 2015
    decade = 2010

if 'Genres' in cdf.columns:
    genres = cdf['Genres'].dropna().astype(str).str.split(',').explode().str.strip()
    top_genre = genres.mode()[0] if not genres.empty else "Indie"
else:
    top_genre = "Indie"

# ─ Tabs Start ─
tab1, tab2, tab3, tab4 = st.tabs(["🎭 Persona", "🌌 Birth Chart", "🔮 Zodiac", "🜁 Elemental"])


# ─🎭 Persona Tab ─
with tab1:
    if val > 0.7 and energy > 0.6:
        vibe = "confident extrovert who dances like nobody’s watching (and doesn’t care if you are)."
    elif val < 0.4 and energy < 0.4:
        vibe = "quiet and introspective, probably journaling under a tree somewhere."
    elif energy > 0.6:
        vibe = "fast-talking, hype-driven, a little chaotic but in a fun way."
    elif val < 0.3:
        vibe = "low-key heartbroken with great taste."
    else:
        vibe = "pretty chill, good with people, but likes alone time too."

    if pop > 70:
        social = "hangs with the popular crowd but is surprisingly down to earth."
    elif pop < 30:
        social = "obscure, misunderstood, possibly lives off-grid and makes killer playlists."
    else:
        social = "comfortable in any room, genre-fluid, probably owns a denim jacket with pins."

    if instr > 0.5:
        hobby = "loves ambient hikes, late-night lo-fi, and probably owns a synth."
    elif speech > 0.5:
        hobby = "obsessed with lyrics, wordplay, and makes epic break-up playlists for friends."
    elif dance > 0.7:
        hobby = "first on the dance floor, last to leave. Probably DJs house parties."
    else:
        hobby = "likes deep cuts, coffee shops, and makes great mixtapes no one else understands."

    st.markdown(f"""
    **If your playlist were a person…**

    - **Born:** ~{year}  
    - **Style:** {top_genre} aesthetic  
    - **Vibe:** {vibe}  
    - **Social Type:** {social}  
    - **Hobbies:** {hobby}
    """)
    st.markdown("---")
    st.subheader("🧬 Persona Name Generator")

    # Core name pools
    genre_core = top_genre.lower()
    name_stems = {
        "pop": ["Nova", "Echo", "Aura", "Raye", "Glow"],
        "rock": ["Ash", "Riot", "Stone", "Vex", "Jett"],
        "hip hop": ["Verse", "Flex", "Dusk", "Shade", "Kruz"],
        "electronic": ["Loop", "Nova", "Flux", "Byte", "Pulse"],
        "folk": ["Hollow", "Bram", "Wren", "Lune", "Moss"],
        "indie": ["Echo", "Field", "Blue", "Sage", "Vale"],
        "metal": ["Crux", "Pyre", "Thorn", "Hex", "Grimm"],
        "jazz": ["Smoke", "Velvet", "Keys", "Miles", "Blue"],
        "r&b": ["Soul", "Velour", "Muse", "Gold", "Rayne"],
        "ambient": ["Drift", "Still", "Nimbus", "Nox", "Vapor"],
    }

    suffix_pool = {
        "🔥 Fire": ["Flare", "Riot", "Wild", "Bolt", "Rush"],
        "🌊 Water": ["Tide", "Flow", "Mist", "Deep", "Wave"],
        "🌪 Air": ["Whirl", "Haze", "Echo", "Current", "Drift"],
        "🌱 Earth": ["Root", "Bloom", "Stone", "Field", "Shade"]
    }

    import random

    # Generate N variations
    def generate_names(n=6):
        base_pool = ["Echo"]
        for g in name_stems:
            if g in genre_core:
                base_pool = name_stems[g]
                break
        suffixes = suffix_pool.get(dominant[0], ["Void"])
        return [f"{random.choice(base_pool)} {random.choice(suffixes)}" for _ in range(n)]

    # Reroll button
    if 'reroll_count' not in st.session_state:
        st.session_state['reroll_count'] = 0

    if st.button("🎲 Generate More Names"):
        st.session_state['reroll_count'] += 1

    names = generate_names()

    st.markdown("Here are some sonic alter egos for your playlist:")
    for name in names:
        st.markdown(f"- `{name}`")


# ─🌌 Birth Chart Tab ─
with tab2:
    if val < 0.35 and energy < 0.4:
        moon = "🌒 Melancholy Moon – Reflective, deep, and emotional"
    elif val > 0.65 and energy > 0.6:
        moon = "🌕 Radiant Moon – Uplifting, bright, and full of joy"
    elif energy > 0.6:
        moon = "🌓 Electric Moon – Energetic and expressive"
    elif val > 0.6:
        moon = "🌔 Sentimental Moon – Warm, nostalgic, and sweet"
    else:
        moon = "🌘 Wandering Moon – Ambiguous, layered, a mystery of moods"

    if decade < 1980:
        rising = "📼 Retro Rising – Soulful, timeless, vintage vibes"
    elif decade < 2000:
        rising = "📻 Analog Rising – Grunge, boom bap, golden age feel"
    elif decade < 2015:
        rising = "🎧 Millennial Rising – Alt, indie, bloghouse and big feelings"
    else:
        rising = "📲 Digital Rising – Fresh, experimental, TikTok-core"

    st.markdown(f"""
    **🌌 Playlist Birth Chart**

    - ☀️ **Sun Sign (Core Genre):** {top_genre}  
    - 🌙 **Moon Sign (Mood):** {moon}  
    - 🌅 **Rising Sign (Decade/Era):** {rising}
    """)

# ─🔮 Zodiac Tab ─
with tab3:
    if val > 0.65 and energy > 0.6:
        zodiac = "🔥 Solar Flare – Radiant, confident, and alive"
    elif val < 0.35 and energy < 0.4:
        zodiac = "🌊 Lunar Drifter – Emotional, floaty, introspective"
    elif tempo_std > 30:
        zodiac = "🌪 Stormchild – Chaotic, intense, unpredictable"
    elif dance > 0.7 and val > 0.55:
        zodiac = "🌈 Joybringer – Danceable, bright, carefree"
    elif acoustic > 0.5 and speech < 0.4:
        zodiac = "🧊 Glassmind – Minimalist and deep-thinking"
    elif val < 0.3 and pop < 50:
        zodiac = "🕳 Voidwalker – Obscure, moody, underground"
    elif decade < 2000:
        zodiac = "🐉 Ancient Pulse – Rooted in past eras"
    elif recent_pct > 0.5:
        zodiac = "🚀 Futurebender – Forward-thinking, fresh"
    else:
        zodiac = "🌀 Fringe Oracle – Genre-resistant, beautifully weird"
        
    st.markdown(f"""
    **🔮 Your Playlist's Music Zodiac**

    - **Zodiac Sign:** {zodiac}
    """)

with tab4:
    # Pull features
    v = cdf['Valence'].mean() if 'Valence' in cdf else 0.5
    e = cdf['Energy'].mean() if 'Energy' in cdf else 0.5
    l = cdf['Loudness'].mean() if 'Loudness' in cdf else -10
    a = cdf['Acousticness'].mean() if 'Acousticness' in cdf else 0.5
    i = cdf['Instrumentalness'].mean() if 'Instrumentalness' in cdf else 0.3
    s = cdf['Speechiness'].mean() if 'Speechiness' in cdf else 0.5
    t_std = cdf['Tempo'].std() if 'Tempo' in cdf else 10
    d = cdf['Danceability'].mean() if 'Danceability' in cdf else 0.5

    # Normalize loudness
    l = (l + 60) / 60  # map -60 to 0 dB → 0 to 1

    # Score each element
    fire = (e + v + l) / 3
    water = (1 - e + 1 - v + a) / 3
    air = (s + d + min(t_std / 50, 1)) / 3
    earth = (a + i + (1 - l)) / 3

    elements = {
        "🔥 Fire": round(fire, 3),
        "🌊 Water": round(water, 3),
        "🌪 Air": round(air, 3),
        "🌱 Earth": round(earth, 3)
    }

    dominant = max(elements.items(), key=lambda x: x[1])

    st.markdown(f"### 🌟 Dominant Element: {dominant[0]}")
    st.markdown({
        "🔥 Fire": "Passionate, expressive, bold. Your playlist kicks down the door.",
        "🌊 Water": "Deep, emotional, immersive. A sonic ocean for soft souls.",
        "🌪 Air": "Clever, quick, curious. A cerebral storm of rhythm and wit.",
        "🌱 Earth": "Grounded, organic, meditative. Built from roots and raw sound."
    }[dominant[0]])

    if st.checkbox("Show full elemental balance", key="elemental_balance_chart"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.barh(list(elements.keys())[::-1], list(elements.values())[::-1], color=["orangered", "deepskyblue", "mediumslateblue", "seagreen"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Elemental Strength")
        ax.set_title("Elemental Breakdown of Playlist")
        st.pyplot(fig)

   
