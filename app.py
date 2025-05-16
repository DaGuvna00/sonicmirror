import streamlit as st

st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎶 SonicMirror – Spotify Playlist Analyzer")

st.write("✅ App is working and running basic Python.")

# This will only show if nothing crashes earlier
try:
    st.write("🔐 Trying to read secrets...")
    st.write("Client ID:", st.secrets["SPOTIPY_CLIENT_ID"][:6] + "...")
except Exception as e:
    st.error(f"❌ Failed to read secrets: {e}")
