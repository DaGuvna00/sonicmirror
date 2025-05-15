import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Your Music Personality Visualizer")

uploaded_file = st.file_uploader("Upload your Exportify CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded! Here's a preview:")
    st.dataframe(df.head())

    # Extract and count genres
    st.subheader("🎧 Top Genres in Your Playlist")
    df['Genres'] = df['Genres'].fillna('').apply(lambda g: [x.strip() for x in g.split(',')])
    genres = pd.Series([genre for sublist in df['Genres'] for genre in sublist])
    top_genres = genres.value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(y=top_genres.index, x=top_genres.values, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Genre")
    ax.set_title("Top Genres")
    st.pyplot(fig)
