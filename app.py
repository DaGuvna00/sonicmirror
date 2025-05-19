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

# ─── Parse and Combine CSVs/Excels ───
if uploaded_files:
    playlists = []
    for f in uploaded_files:
        name = f.name.rsplit('.',1)[0]
        try:
            if f.name.lower().endswith('.csv'):
                df = pd.read_csv(f)
            else:
                sheets = pd.read_excel(f, sheet_name=None)
                df = pd.concat(sheets.values(), ignore_index=True)
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
        st.error("No valid files uploaded. Please check your Exportify export format.")
        st.stop()
    data = pd.concat(playlists, ignore_index=True)
else:
    st.info("📥 Upload at least one Exportify file to begin analysis.")
    st.stop()

# ─── Sidebar Controls ───
st.sidebar.header("🔎 Analysis Options")
playlist_names = data['Playlist'].unique().tolist()
selected = st.sidebar.multiselect("Choose playlists to include", playlist_names, default=playlist_names)
features = ['Energy','Valence','Danceability','Acousticness','Instrumentalness','Liveness','Speechiness','Tempo','Loudness']
selected_feats = st.sidebar.multiselect("Select audio features", features, default=features)

# ─── Date Parsing & Lag Calculation ───
# Convert to UTC then drop timezone so both columns align
data['AddedAt'] = pd.to_datetime(data['AddedAt'], errors='coerce', utc=True).dt.tz_convert(None)
data['ReleaseDate'] = pd.to_datetime(data['ReleaseDate'], errors='coerce', utc=True).dt.tz_convert(None)
# Compute discovery lag in days
data['LagDays'] = (data['AddedAt'] - data['ReleaseDate']).dt.days

# ─── Filter & Prepare Dashboard Data ───
df = data[data['Playlist'].isin(selected)].copy()

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
    st.image(wc.to_array(), use_column_width=True)

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

# ─── Playlist Similarity Matrix ───
st.header("🗂️ Playlist Similarity")
# Compute cosine similarity between playlists based on selected_feats
from numpy.linalg import norm
# Build feature matrix
X = avgs[selected_feats].values
labels = avgs.index.tolist()
# Normalize rows
norms = norm(X, axis=1, keepdims=True)
X_norm = X / norms
# Cosine similarity matrix
sim_matrix = np.dot(X_norm, X_norm.T)
sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
# Display similarity table
st.subheader("Cosine Similarity Table")
st.dataframe(sim_df.round(3))
# Heatmap
fig_sim, ax_sim = plt.subplots()
cax = ax_sim.matshow(sim_df, vmin=0, vmax=1)
fig_sim.colorbar(cax)
ax_sim.set_xticks(range(len(labels)))
ax_sim.set_yticks(range(len(labels)))
ax_sim.set_xticklabels(labels, rotation=90)
ax_sim.set_yticklabels(labels)
ax_sim.set_title('Playlist Cosine Similarity')
st.pyplot(fig_sim)

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
    metrics = selected_feats
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

# ─── Festival Poster Generator ───import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os

# ─── Festival Poster Generator ───
st.header("🎪 Downloadable Festival Poster")

if 'Artist' in df.columns:
    from collections import Counter

    all_artists = []
    for a in df['Artist'].dropna():
        all_artists.extend([x.strip() for x in str(a).split(',')])

    artist_counts = Counter(all_artists)
    top_40 = artist_counts.most_common(40)  # increase from 20 to 40
    top_artists = [a for a, _ in top_40]

    # Split into two days
    midpoint = len(top_artists) // 2
    day_1 = top_artists[:midpoint]
    day_2 = top_artists[midpoint:]

    def build_poster(day1, day2):
        fig, ax = plt.subplots(figsize=(10, 16))  # increased size to fit more
        ax.axis('off')

    # Load background image
    bg_path = "festival_background.jpg"
    if os.path.exists(bg_path):
        bg_img = Image.open(bg_path)
        ax.imshow(bg_img, extent=[0, 1, 0, 1], aspect='auto', zorder=0)
    else:
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#121212")

    # Draw glowing title text
    for dx, dy in [(-0.002, -0.002), (0.002, -0.002), (-0.002, 0.002), (0.002, 0.002)]:
        ax.text(0.5 + dx, 0.96 + dy, "SONICMIRROR FESTIVAL", fontsize=28, fontweight='bold',
                ha='center', color='black', zorder=1, family='monospace')
    ax.text(0.5, 0.96, "SONICMIRROR FESTIVAL", fontsize=28, fontweight='bold',
            ha='center', color='white', zorder=1, family='monospace')

    # Draw artist lineup for each day
    def draw_day(title, artists, y_start):
        ax.text(0.5, y_start, title, fontsize=24, fontweight='bold',
                ha='center', color='white', zorder=1, family='monospace')
        ax.text(0.5, y_start - 0.05, artists[0], fontsize=34, fontweight='bold',
                ha='center', color='gold', zorder=1, family='monospace')
        ax.text(0.5, y_start - 0.10, ' • '.join(artists[1:4]), fontsize=18,
                ha='center', color='white', zorder=1, family='monospace')
        ax.text(0.5, y_start - 0.15, ' • '.join(artists[4:8]), fontsize=14,
                ha='center', color='lightgray', zorder=1, family='monospace')
        ax.text(0.5, y_start - 0.22,
                '\n'.join([' • '.join(artists[i:i+6]) for i in range(8, len(artists), 6)]),
                fontsize=12, ha='center', color='lightgray', zorder=1, family='monospace')

    draw_day("DAY 1", day1, 0.88)
    draw_day("DAY 2", day2, 0.48)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
return buf

