import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import os

# Set up Streamlit page
st.set_page_config(page_title="SonicMirror", layout="wide")
st.title("🎵 SonicMirror – Spotify Playlist Analyzer")

# Try to use session token to authenticate
if "access_token" not in st.session_state:
    st.warning("🔒 Please log in with Spotify to continue:")
    auth_manager = SpotifyOAuth(
        client_id=st.secrets["SPOTIPY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
        scope="playlist-read-private playlist-read-collaborative"
    )
    auth_url = auth_manager.get_authorize_url()
    st.markdown(f"[Click here to log in with Spotify]({auth_url})")

    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        code = query_params["code"][0]
        token_info = auth_manager.get_access_token(code, as_dict=False)
        st.session_state.access_token = token_info['access_token']
        st.experimental_rerun()
else:
    # Logged in
    sp = spotipy.Spotify(auth=st.session_state.access_token)
    user = sp.current_user()
    st.success(f"📅 Logged in as: {user['display_name']}")

    # ---- Playlist Selection ----
    playlists = sp.current_user_playlists()['items']
    playlist_names = [p['name'] for p in playlists]
    playlist_map = {p['name']: p['id'] for p in playlists]

    st.subheader("🎶 Select Playlists to Analyze")
    selected_names = st.multiselect("Choose one or more playlists", playlist_names)

    # ---- If selected and button clicked ----
    if selected_names and st.button("Analyze Selected Playlists"):
        st.write("▶️ Analyze button clicked")
        all_tracks = []
        st.write("🌟 Tracks found:", len(all_tracks))

        for name in selected_names:
            playlist_id = playlist_map[name]
            offset = 0

            while True:
                results = sp.playlist_tracks(playlist_id, offset=offset, limit=100)
                tracks = results['items']
                if not tracks:
                    break

                for item in tracks:
                    track = item['track']
                    if track and track["id"]:
                        all_tracks.append({
                            "playlist": name,
                            "name": track["name"],
                            "id": track["id"],
                            "artists": ", ".join([a["name"] for a in track["artists"]]),
                            "album": track["album"]["name"]
                        })

                offset += 100

        def get_audio_features_in_batches(track_ids):
            audio_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                try:
                    features = sp.audio_features(batch)
                    audio_features.extend(features)
                except Exception as e:
                    st.error(f"🚫 Error fetching batch {i//100+1}: {e}")
            return audio_features

        track_ids = [t["id"] for t in all_tracks if t["id"]]
        audio_features = get_audio_features_in_batches(track_ids)
        st.write("🎵 Got audio features for", len(track_ids), "tracks.")

        for i, features in enumerate(audio_features):
            all_tracks[i].update(features)

        df = pd.DataFrame(all_tracks)
        st.write("📊 DataFrame shape:", df.shape)
        st.dataframe(df[["playlist", "name", "artists", "energy", "valence", "danceability", "tempo"]])
        st.session_state.df = df
