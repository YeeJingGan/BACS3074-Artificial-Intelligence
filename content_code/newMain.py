import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# (can remove) function to create the preprocessed data csv file
def preprocess_data():
    # a) Load data
    df_books = pd.read_csv('csv/Books.csv')
    df_ratings = pd.read_csv('csv/Ratings.csv')

    # b) Data cleaning step 1: Fix format issue in year of publication(df_books) and convert to int data type
    authors = []
    books_titles = []
    temp = df_books[(df_books['Year-Of-Publication'] == 'DK Publishing Inc') | (df_books['Year-Of-Publication'] == 'Gallimard')]
    error_indexes = pd.Series(list(temp.index))

    for title in temp['Book-Title']:
        author = title.split(';')[-1].split('"')[0]
        book = title.split(';')[0].split('\\')[0]
        authors.append(author)
        books_titles.append(book)
    temp = pd.concat([temp['ISBN'].to_frame(), temp[df_books.columns[1:]].shift(periods=1, axis=1)], axis=1)
    temp['Book-Title'] = books_titles
    temp['Book-Author'] = authors
    df_books.drop(error_indexes, axis=0, inplace=True)

    for i in error_indexes:
        error_indexes.loc[i] = list(temp.loc[i].values)
    df_books['Year-Of-Publication'] = df_books['Year-Of-Publication'].astype(int)
    #df_books.drop(labels=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)

    # c) Data Cleaning step 2 : Only keep credible users
    f = ['count', 'mean']
    df_books_summary = df_ratings.groupby('ISBN')['Book-Rating'].agg(f)
    df_books_summary.index = df_books_summary.index.map(str)
    drop_book_list = df_books_summary[df_books_summary['count'] < 10].index

    df_users_summary = df_ratings.groupby('User-ID')['Book-Rating'].agg(f)
    df_users_summary.index = df_users_summary.index.map(int)
    drop_users_list = df_users_summary[df_users_summary['count'] < 10].index

    df_ratings = df_ratings[~df_ratings['ISBN'].isin(drop_book_list)]
    df_ratings = df_ratings[~df_ratings['User-ID'].isin(drop_users_list)]

    # d) Preprocess data for model input
    features = ['Book-Author', 'Year-Of-Publication', 'Publisher']
    df_books_preprocess = df_books[df_books['ISBN'].isin(df_ratings['ISBN'].unique())].copy()
    for feature in features:
        if df_books_preprocess[feature].dtype == 'O':
            df_books_preprocess[feature] = df_books_preprocess[feature].str.replace('\W', '', regex=True)
            df_books_preprocess[feature] = df_books_preprocess[feature].apply(lambda x: str.lower(x))

    # Save preprocessed data to CSV files
    df_books_preprocess.to_csv('preprocessed_books.csv', index=False)
    df_ratings.to_csv('preprocessed_ratings.csv', index=False)

    return df_books_preprocess, df_ratings, df_books

# cbRecSystem - cosine similarity (But if user rate less than 3 books, then get most popular books)
def get_recommendations(userid):
    # Load preprocessed data
    df_books_preprocess = pd.read_csv(r'C:\Users\hp\Downloads\ai_assignment\content_code\preprocessed_books.csv')
    df_ratings = pd.read_csv(r'C:\Users\hp\Downloads\ai_assignment\content_code\preprocessed_ratings.csv')
    df_books = pd.read_csv(r'C:\Users\hp\Downloads\ai_assignment\csvs\Books.csv')

    # Calculate the most popular books
    popular_books = df_ratings.groupby('ISBN')['Book-Rating'].count().sort_values(ascending=False)

    def get_popular_books(num_books=10):
        # Returns a DataFrame with the most popular books
        popular_books_df = popular_books.reset_index()[['ISBN']][:num_books]
        return df_books[df_books['ISBN'].isin(popular_books_df['ISBN'])]

    # Check if the user has rated fewer than 3 books
    if df_ratings[df_ratings['User-ID'] == userid]['Book-Rating'].count() < 3:
        # Return the most popular books
        return get_popular_books()

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(
        df_books_preprocess['Book-Author'] + ' ' + df_books_preprocess['Year-Of-Publication'].astype(str) + ' ' + df_books_preprocess['Publisher']
    )
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    df2 = df_books_preprocess.reset_index()
    indices = pd.Series(df2.index, index=df2['ISBN'])

    def recommend_books(ISBN, cosine_sim):
        idx = indices[ISBN]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        book_indices = [i[0] for i in sim_scores]
        return df_books.iloc[book_indices]

    ISBN = df_ratings['ISBN'].loc[df_ratings[df_ratings['User-ID'] == userid]['Book-Rating'].idxmax()]
    recommendation = recommend_books(ISBN, cosine_sim)

    return recommendation