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
