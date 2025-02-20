import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import SVD
import warnings
import streamlit as st

warnings.filterwarnings("ignore")


def get_general_recommendations():

    # 1. Group books by isbn
    grouped_ratings = st.session_state['yj_df_ratings'].groupby('ISBN')

    # 2. Count how many ratings given to each books
    ratings_count = grouped_ratings.size()

    # 3. Sort books by the number of ratings in descending order
    ratings_count = ratings_count.sort_values(ascending=False)

    # 4. Get the top 10 books with the most ratings
    top_books = ratings_count.head(10)

    # 5. Extract info of top 10 books
    df_recommended_books = st.session_state['yj_df_books'][st.session_state['yj_df_books']['ISBN'].isin(top_books.index)]

    return df_recommended_books


def get_recommendations(user_id, predictions = None):

    # 1. Find out all predictions made to this user
    ratings_prediction = [pred for pred in st.session_state['yj_predictions_svd'] if pred.uid == user_id]
    if predictions is not None:
        ratings_prediction = [pred for pred in predictions if pred.uid == user_id]

    # 2. Sort in descending order from highest to lowest
    sorted_predictions = sorted(ratings_prediction, key=lambda x: x.est, reverse = True)
    # 3. Extract top 10 books ISBN
    top_10_books_isbn = [pred.iid for pred in sorted_predictions [:10]]
    # 4. Return books info for these 10 books
    df_recommended_books = st.session_state['yj_df_books'][st.session_state['yj_df_books']['ISBN'].isin(top_10_books_isbn)]

    return df_recommended_books


def update_recommendations(df_new, user_id):
    # 1. Concatenate datasets
    df_concatenated = pd.concat([st.session_state['yj_cleaned_dataset'], df_new], axis = 0)
    df_concatenated.reset_index(drop = True, inplace = True)

    # 2. Convert to Surprise dataset
    concatenated_dataset = Dataset.load_from_df(df_concatenated[['User-ID','ISBN','Book-Rating']], st.session_state['yj_reader'])

    # 3. Perform train test split
    train_set, test_set = train_test_split(concatenated_dataset, test_size = 0.2)

    # 4. Retrain model
    predictions_svd = SVD().fit(train_set).test(test_set)

    return get_recommendations(user_id, predictions_svd)


def main():

    if 'yj_df_books' not in st.session_state:
        st.session_state['yj_df_books'] = pd.read_csv(r'C:\Users\hp\Downloads\ai_assignment\csvs\Books.csv')

    if 'yj_df_ratings' not in st.session_state:
        df_ratings = pd.read_csv(r'C:\Users\hp\Downloads\ai_assignment\csvs\Ratings.csv')
        st.session_state['yj_df_ratings'] = df_ratings[df_ratings['Book-Rating'] != 0]

    if 'yj_reader' not in st.session_state:
        st.session_state['yj_reader'] = Reader(rating_scale=(1, 10))

    if 'yj_cleaned_dataset' not in st.session_state:
        st.session_state['yj_cleaned_dataset'] = pd.read_csv(r'C:\Users\hp\Downloads\ai_assignment\cf_code\cf_final.csv')

    if 'yj_final_dataset' not in st.session_state:
        st.session_state['yj_final_dataset'] = Dataset.load_from_df(st.session_state['yj_cleaned_dataset'], st.session_state['yj_reader'])

    # Split according to 8:2 ratio
    if 'yj_train_set' not in st.session_state or 'yj_test_set' not in st.session_state:
        st.session_state['yj_train_set'], st.session_state['yj_test_set'] = train_test_split(st.session_state['yj_final_dataset'], test_size = 0.2)

    if 'yj_svd' not in st.session_state:
        st.session_state['yj_svd'] = SVD()

    if 'yj_predictions_svd' not in st.session_state:
        st.session_state['yj_predictions_svd'] = st.session_state['yj_svd'].fit(st.session_state['yj_train_set']).test(st.session_state['yj_test_set'])


if __name__ == '__main__':
    main()