
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile

st.title("Movie Recommender")

@st.cache_data
def load_data():
    
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags    = pd.read_csv('tags.csv')
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()
    avg_ratings = ratings.groupby('movieId')['rating'].agg(rating='mean', num_ratings='count').round(2).reset_index()
    movie_tags  = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index().rename(columns={'tag':'tags'})
    df = movies.merge(avg_ratings, on='movieId', how='left')
    df = df.merge(movie_tags, on='movieId', how='left')
    df['rating']      = df['rating'].fillna(0.0)
    df['num_ratings'] = df['num_ratings'].fillna(0).astype(int)
    df['tags']        = df['tags'].fillna('')
    df['soup']        = df['genres_clean'] + ' ' + df['tags']
    return df

@st.cache_data
def build_sim_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['soup'])
    return cosine_similarity(matrix, matrix)

df = load_data()
sim_matrix = build_sim_matrix(df)

if 'liked' not in st.session_state:
    st.session_state.liked = []

# Sidebar
st.sidebar.header("Filters")
all_genres = sorted(set(g for genres in df['genres'].dropna() for g in genres.split('|') if g != '(no genres listed)'))
selected_genres = st.sidebar.multiselect("Genres", all_genres)
min_r, max_r    = st.sidebar.slider("Rating", 0.0, 5.0, (3.0, 5.0), step=0.5)
min_votes       = st.sidebar.slider("Min votes", 0, 500, 50, step=10)

filtered_df = df.copy()
if selected_genres:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda g: any(x in str(g) for x in selected_genres))]
filtered_df = filtered_df[filtered_df['rating'].between(min_r, max_r)]
filtered_df = filtered_df[filtered_df['num_ratings'] >= min_votes].reset_index(drop=True)
st.sidebar.write(f"{len(filtered_df)} movies found")

# Tabs
tab1, tab2 = st.tabs(["Recommend", "Liked List"])

with tab1:
    movie = st.selectbox("Pick a movie", sorted(filtered_df['title'].dropna().unique()))
    n     = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Get recommendations"):
        idx    = df[df['title'] == movie].index[0]
        scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)
        scores = [(i, s) for i, s in scores if i != idx][:n]

        recs = df.iloc[[i[0] for i in scores]][['title','genres','rating']].copy()
        recs['similarity'] = [round(s[1], 3) for s in scores]
        recs.sort_values(['similarity','rating'], ascending=[False,False], inplace=True)
        recs.reset_index(drop=True, inplace=True)

        st.dataframe(recs, use_container_width=True)

        for _, row in recs.iterrows():
            c1, c2 = st.columns([5, 1])
            c1.write(f"{row['title']} ({row['genres']})")
            if c2.button("Like", key=f"like_{row['title']}"):
                if row['title'] not in st.session_state.liked:
                    st.session_state.liked.append(row['title'])
                    st.toast(f"Added: {row['title']}")

with tab2:
    if not st.session_state.liked:
        st.info("No liked movies yet!")
    else:
        for m in st.session_state.liked:
            c1, c2 = st.columns([4, 1])
            c1.write(m)
            if c2.button("Remove", key=f"remove_{m}"):
                st.session_state.liked.remove(m)
                st.rerun()
        if st.button("Clear all"):
            st.session_state.liked = []
            st.rerun()
