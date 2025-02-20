import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import keras
from math import sqrt
import streamlit as st
import pickle

sys.path.append(r'C:\Users\hp\Downloads\ai_assignment')

# Establish connection to database
try:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\ai_assignment\firebase\ai-assignment-2885a-firebase-adminsdk-ls4zo-1155c285b4.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

# Reading collaborative dataframe from csv
if 'js_standard_scaler' not in st.session_state:
    st.session_state['js_standard_scaler'] = StandardScaler()
if 'js_df_collaborative' not in st.session_state:
    st.session_state['js_df_collaborative'] = pd.read_csv(r'hybrid_code\hybrid_csvs\df_collaborative.csv')
    st.session_state['js_df_collaborative']['Book-Rating-Scaled'] = st.session_state['js_standard_scaler'].fit_transform(st.session_state['js_df_collaborative'][['Book-Rating']])
if 'js_df_collaborative_pivot' not in st.session_state:
    st.session_state['js_df_collaborative_pivot'] = st.session_state['js_df_collaborative'].pivot_table(index='User-ID', columns='ISBN',values='Book-Rating-Scaled').fillna(0)

# Reading final combined csv
if 'js_df_final' not in st.session_state:
    st.session_state['js_df_final'] = pd.read_csv(r'hybrid_code\hybrid_csvs\df_final.csv')
if 'js_df_books' not in st.session_state:
    st.session_state['js_df_books'] = pd.read_csv(r'csvs\Books.csv')

# Reading u, sigma, vt
if 'js_u' not in st.session_state:
    st.session_state['js_u'] = np.load(r'hybrid_code\svds\u.npy')
if 'js_sigma' not in st.session_state:
    st.session_state['js_sigma'] = np.load(r'hybrid_code\svds\sigma.npy')
if 'js_vt' not in st.session_state:
    st.session_state['js_vt'] = np.load(r'hybrid_code\svds\vt.npy')

# Retrieving TFIDF Vectorizer object
if 'js_tfidf_vectozizer' not in st.session_state:
    with open(r'hybrid_code\embeddings\tfidf_vectorizer.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    st.session_state['js_tfidf_vectozizer'] = tfidf_vectorizer

# Storing pretrained model in session_state
if 'js_keras_model' not in st.session_state:
    st.session_state['js_keras_model'] = keras.models.load_model(r'hybrid_code\keras\mlp_model_2.keras')

scaler = st.session_state['js_standard_scaler']
df_collaborative = st.session_state['js_df_collaborative']
scaler_mean = scaler.mean_[0]
scaler_std = scaler.scale_[0]
df_collaborative_pivot = st.session_state['js_df_collaborative_pivot']

df_final = st.session_state['js_df_final']
df_book = st.session_state['js_df_books']

u = st.session_state['js_u']
sigma = st.session_state['js_sigma']
vt = st.session_state['js_vt']

tfidf_vectorizer = st.session_state['js_tfidf_vectozizer']
keras_model = st.session_state['js_keras_model']

# ========================================================Collaborative Based Recommender System===================================================================
# Function to get target user interaction in pandas.Series|None based on user_id
def get_target_user_interaction(user_id):
    df_database_interaction = pd.DataFrame()

    query_result = db.collection('ratings').where("user_id", "==", user_id).get()

    if query_result:
        new_data = []
        for doc in query_result:
            new_data.append(doc.to_dict())

        df_database_interaction = pd.DataFrame(new_data).rename(columns={"user_id": "User-ID",
                                                                     "isbn": "ISBN",
                                                                     "book_rating": "Book-Rating"})

        df_database_interaction.sort_values(by='Book-Rating', ascending=False,inplace=True)
        df_database_interaction.drop(columns=['User-ID'],inplace=True)
        df_database_interaction.drop_duplicates(subset=['ISBN'],keep='first',inplace=True)
        df_database_interaction = df_database_interaction[df_database_interaction['ISBN'].isin(df_collaborative_pivot.columns)].rename(columns={'ISBN': 'ISBN', 'Book-Rating': 'Database Rating'})
        df_database_interaction['Database Rating'] = df_database_interaction['Database Rating'].apply(lambda x: (x - scaler_mean) / scaler_std)

    # Getting past interaction of target user if found in the pivot table
    df_past_interaction = pd.DataFrame()
    if user_id in df_collaborative_pivot.index:
        df_past_interaction = df_collaborative_pivot.loc[user_id].reset_index().rename(columns={'ISBN':'ISBN',user_id:'Past Rating'})


    # Getting other interaction of target user having related ISBNs found in the pivot table, if found in combined dataframe
    df_other_interaction = pd.DataFrame()
    if user_id in df_final['User-ID']:
        df_other_interaction = df_final[(df_final['ISBN'].isin(df_collaborative_pivot.columns)) & (df_final['User-ID'] == user_id)][['ISBN','Book-Rating']].rename(columns={'ISBN':'ISBN','Book-Rating':'Other Rating'})
        df_other_interaction['Other Rating'] = df_other_interaction['Other Rating'].apply(lambda x: (x - scaler_mean) / scaler_std)


    # Combining all sources of interaction into a dataframe with ISBNs found in pivot table as one column
    df_all_interaction = pd.DataFrame()
    df_all_interaction['ISBN'] = df_collaborative_pivot.columns
    df_all_interaction['Default Rating'] = (0 - scaler_mean) / scaler_std

    if not df_past_interaction.empty:
        df_all_interaction = df_past_interaction.copy()
    if not df_other_interaction.empty:
        df_all_interaction = pd.merge(df_all_interaction, df_other_interaction, how='left', on='ISBN')
        df_all_interaction = df_all_interaction.fillna((0 - scaler_mean) / scaler_std)
    if not df_database_interaction.empty:
        df_all_interaction = pd.merge(df_all_interaction, df_database_interaction, how='left', on='ISBN')
        df_all_interaction = df_all_interaction.fillna((0 - scaler_mean) / scaler_std)

    # Finding maximum value in each type of interaction with the ISBN
    if not df_past_interaction.empty:
        df_all_interaction['Max'] = df_all_interaction['Past Rating'].values
        if not df_other_interaction.empty:
            df_all_interaction['Max'] = df_all_interaction['Past Rating'].where(df_all_interaction['Past Rating'] > df_all_interaction['Other Rating'], df_all_interaction['Other Rating'])
        if not df_database_interaction.empty:
            df_all_interaction['Max'] = df_all_interaction['Max'].where(df_all_interaction['Max'] > df_all_interaction['Database Rating'], df_all_interaction['Database Rating'])
    elif not df_other_interaction.empty:
        df_all_interaction['Max'] = df_all_interaction['Other Rating'].values
        if not df_database_interaction.empty:
            df_all_interaction['Max'] = df_all_interaction['Other Rating'].where(df_all_interaction['Other Rating'] > df_all_interaction['Database Rating'], df_all_interaction['Database Rating'])
    elif not df_database_interaction.empty:
        df_all_interaction['Max'] = df_all_interaction['Database Rating'].values
    else:
        return None

    # Returning the maximum interaction with ISBN set as index
    df_all_interaction = df_all_interaction.set_index('ISBN')
    return df_all_interaction['Max']

# Function to get n nearest neighbours in pandas.Dataframe based on cosine similarity
def find_n_nearest_neighbours_with_cosine_similarity(user_id, target_user_interaction, n=7):
    neighbours_distance = pd.Series([1 - cosine(target_user_interaction, df_collaborative_pivot.loc[neighbour_id]) for neighbour_id in df_collaborative_pivot.index])
    neighbours_distance.name = 'Cosine Similarity'

    # Returning index 1 to n+1 if user_id is found in the pivot table else 0 to n
    add = 1 if user_id in df_collaborative_pivot.index else 0
    return pd.concat([neighbours_distance, pd.Series(df_collaborative_pivot.index)], axis=1).sort_values(by='Cosine Similarity', ascending=False)[add : n + add]

# Function to get recommendations in pandas.Series given a user_id
def get_recommendations(user_id):
    user_latent_factors = u[df_collaborative_pivot.index.get_loc(user_id)]
    predicted_ratings = np.dot(user_latent_factors, np.dot(np.diag(sigma), vt))
    predicted_ratings_scaled = scaler.inverse_transform(predicted_ratings.reshape(-1, 1))[:, 0]
    return pd.Series(predicted_ratings_scaled, index=df_collaborative_pivot.columns).sort_values(ascending=False)

# Function to get all neighbours recommendations in pandas.Dataframe
def get_all_neighbours_recommendations(neighbour_ids):
    all_neighbours_recommendations = pd.DataFrame()
    for neighbour_id in neighbour_ids:
        neighbour_recommendations = get_recommendations(neighbour_id)
        neighbour_recommendations.name = neighbour_id
        all_neighbours_recommendations = pd.concat([all_neighbours_recommendations,neighbour_recommendations], axis=1)
    return all_neighbours_recommendations

# Function to get weighted average recommendations in pandas.Series
def get_collaborative_weighted_recommendations(df_neighbours,df_all_neighbour_recommendations):
    df_all_with_weights = pd.merge(df_neighbours, df_all_neighbour_recommendations.stack().reset_index(name = 'Predicted Rating').rename(columns = {'level_0':'ISBN','level_1':'User-ID'}), on = 'User-ID')
    df_all_with_weights['Weighted Sum'] = df_all_with_weights['Cosine Similarity'] * df_all_with_weights['Predicted Rating']
    return df_all_with_weights.groupby('ISBN')['Weighted Sum'].sum() / df_all_with_weights.groupby('ISBN')['Cosine Similarity'].sum()

# Function to get collaborative recommendations in pandas.Series and mean of cosine distance from the target user's neighbours
def get_collaborative_recommendations(user_id, n = 7):
    target_user_interaction = get_target_user_interaction(user_id)
    if target_user_interaction is not None:
        df_neighbours = find_n_nearest_neighbours_with_cosine_similarity(user_id, target_user_interaction, n = n)
        df_all_neighbour_recommendations = get_all_neighbours_recommendations(df_neighbours['User-ID'])
        return get_collaborative_weighted_recommendations(df_neighbours, df_all_neighbour_recommendations), df_neighbours['Cosine Similarity'].mean()
    else:
        return None, 0

# ========================================================Content Based Recommender System===================================================================
# Function to get location of target user in str
def get_user_location(user_id):
    query_result = db.collection('users').where("user_id", "==", user_id).get()

    if query_result:
        return query_result[0].to_dict()['location']
    elif user_id in df_final['User-ID']:
        return df_final[df_final['User-ID'] == user_id ]['Location'].iloc[0]
    else:
        return 'n/a, n/a, n/a'

# Function to get embeddings in pandas.Dataframe
def get_embedded_features(user_location):
    df_target = df_final.drop_duplicates(subset=['ISBN'])
    df_target_text_features = df_target[['Book-Title','Book-Author','Publisher']]
    df_target_text_features['Location'] = user_location
    df_target_text_features = df_target_text_features[['Book-Title','Book-Author','Publisher','Location']].apply(lambda x: ' '.join(x), axis = 1)
    df_target_tfidf = tfidf_vectorizer.fit_transform(df_target_text_features)

    df_target_features = pd.DataFrame(df_target_tfidf.todense(),columns=tfidf_vectorizer.get_feature_names_out())
    df_target_features['Year-Of-Publication'] = df_target['Year-Of-Publication'].values
    return df_target_features

# Function to get content recommendations in pandas.Series
def get_content_recommendations(user_id):
    if 'js_content_recommendations' in st.session_state:
        return st.session_state['js_content_recommendations']

    best_mlp_model = keras_model
    location = get_user_location(user_id)
    df_embedded_features = get_embedded_features(location)

    predictions = best_mlp_model.predict(df_embedded_features)
    predictions = pd.Series(predictions.flatten())
    predictions.name = 'MLP Predicted Rating'
    predictions.index = df_final.drop_duplicates(subset=['ISBN'])['ISBN']

    st.session_state['js_content_recommendations'] = predictions
    return predictions

# ========================================================Content Based Recommender System===================================================================
# Function to get hybrid recommendations in pandas.Dataframe
def get_hybrid_recommendations(user_id):
    collaborative_recommendations, average_cosine_similarity = get_collaborative_recommendations(user_id)
    content_recommendations = get_content_recommendations(user_id)

    if collaborative_recommendations is not None:
        df_merged_recommendations = pd.merge(content_recommendations.reset_index(), collaborative_recommendations.reset_index(), how='left', on='ISBN').fillna(scaler_mean)
        df_merged_recommendations = df_merged_recommendations.rename(columns={0: 'SVD Predicted Rating'})

        weighting_factor = sqrt(average_cosine_similarity)
        df_merged_recommendations['Weighted Average'] = (df_merged_recommendations['SVD Predicted Rating'] * weighting_factor) + (df_merged_recommendations['MLP Predicted Rating'] * (1 - weighting_factor))
    else:
        df_merged_recommendations = pd.DataFrame(content_recommendations.reset_index())
        df_merged_recommendations['Weighted Average'] = df_merged_recommendations['MLP Predicted Rating']

    df_merged_recommendations = df_merged_recommendations.sort_values(by='Weighted Average', ascending=False)[:10]

    df_return = df_book[df_book['ISBN'].isin(df_merged_recommendations['ISBN'])]

    return df_return