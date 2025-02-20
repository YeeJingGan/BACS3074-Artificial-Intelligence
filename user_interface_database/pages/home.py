import time
import firebase_admin
import pandas as pd
import streamlit as st
import sys
from firebase_admin import credentials, firestore

sys.path.append(r'C:\Users\hp\Downloads\ai_assignment')
from hybrid_code import hybrid_recommender_system as hybrid_model
from cf_code import cf_recommender_system as cf_model
from content_code import newMain as content_model

# Establish connection to database
try:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\ai_assignment\firebase\ai-assignment-2885a-firebase-adminsdk-ls4zo-1155c285b4.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

# Check user ID
# sample user:
# st.session_state['existing_user_id'] = 76499

# Check session state
if 'rate_error' not in st.session_state:
    st.session_state['rate_error'] = ''
if 'rate_success' not in st.session_state:
    st.session_state['rate_success'] = ''
if 'newly_rated_book_count' not in st.session_state:
    st.session_state['newly_rated_book_count'] = 0
if 'refresh_btn' not in st.session_state:
    st.session_state['refresh_btn'] = True

# Welcome header
welcome_message = f"<h3>Welcome user {st.session_state['existing_user_id']}!</h3>"
st.markdown(welcome_message, unsafe_allow_html=True)

# Sidebar to choose model
sidebar = st.sidebar
model_choice = sidebar.selectbox("Please select one (1) model",
                                    ('Content-Based', 'Collaborative Filtering-Based', 'Hybrid-Based'))

# Select Box to choose book
if 'df_books' not in st.session_state:
    st.session_state['df_books'] = pd.read_csv(r'C:\Users\hp\Downloads\ai_assignment\csvs\Books.csv')

selected_book = st.selectbox("Select a Book: ", st.session_state['df_books']['Book-Title'], index=None)


# Function to create messages for respective rating made by user
def validate_rating():
    print(st.session_state.rating)
    if st.session_state.rating is not None:
        st.session_state['rate_success'] = "You have successfully rated book " + str(st.session_state.isbn) + " with Rating " + str(st.session_state.rating)
    else:
        st.session_state['rate_error'] = "You did not provide rating for book " + str(st.session_state.isbn)


# Display success or error message and store rating data to database
if st.session_state['rate_success'] != '':
    # Write rating to database
    rating_data = {
        'user_id': st.session_state['existing_user_id'],
        'isbn': st.session_state.isbn,
        'book_rating': st.session_state.rating
    }
    db.collection("ratings").add(rating_data)

    # Increase rating count made by this user in current session
    if model_choice == 'Collaborative Filtering-Based':
        st.session_state['newly_rated_book_count'] += 1

    # Make refresh recommendations button available if count equals 3
    if model_choice == 'Collaborative Filtering-Based' and st.session_state['newly_rated_book_count'] == 3:
        st.session_state['refresh_btn'] = False
        st.session_state['newly_rated_book_count'] = 0  # Reset count

    # Display success message
    success = st.success(st.session_state['rate_success'])
    time.sleep(5)
    success.empty()
    del st.session_state['rate_success']
    del st.session_state['rate_error']

elif st.session_state['rate_error'] != '':

    # Display error message if rating not given properly
    error = st.error(st.session_state['rate_error'])
    time.sleep(5)
    error.empty()
    del st.session_state['rate_error']
    del st.session_state['rate_success']


# Function to check if new ratings more than 3 to update recommendations (Used only at the beginning of each session)
# (Only use for cf)
def check_update_needed():
    # Query to get all ratings made by this user
    query_result = db.collection('ratings').where("user_id", "==", st.session_state['existing_user_id']).get()

    new_data = []
    document_count = 0

    # Retrieve rating records and count how many ratings has made
    for doc in query_result:
        document_count += 1
        new_data.append(doc.to_dict())

    # Convert rating records to dataframe
    df_new_data = pd.DataFrame(new_data).rename(columns={"user_id": "User-ID",
                                                         "isbn": "ISBN",
                                                         "book_rating": "Book-Rating"})

    # Indicate recommendations need to be updated if count more than 3
    if document_count >= 3:
        st.session_state.df_new_data = df_new_data
        return True
    else:
        return False


# Generate recommendations based on model choose
if model_choice == 'Content-Based':
    st.session_state['refresh_btn'] = True
    st.session_state['newly_rated_book_count'] = 0
    content_start_time = time.time();
    if 'jr_content_recommendations' not in st.session_state:
        st.session_state['jr_content_recommendations'] = content_model.get_recommendations(st.session_state['existing_user_id'])
    content_end_time = time.time();
    content_duration = content_end_time - content_start_time;
    sidebar.write("Runtime: " + str(content_duration) + " s")

elif model_choice == 'Collaborative Filtering-Based':
    cf_start_time = time.time();
    if 'yj_cf_recommendations' not in st.session_state:
        st.session_state['refresh_btn'] = True
        cf_model.main()

        # Check if no recommendations generated
        recommendations = cf_model.get_recommendations(st.session_state['existing_user_id'])
        if recommendations.empty:
            # Display general recommendations
            st.session_state['yj_cf_recommendations'] = cf_model.get_general_recommendations()
        else:
            # Check if recommendations need to be updated
            if check_update_needed():
                st.session_state['yj_cf_recommendations'] = cf_model.update_recommendations(st.session_state.df_new_data, st.session_state['existing_user_id'])
            else:
                st.session_state['yj_cf_recommendations'] = cf_model.get_recommendations(st.session_state['existing_user_id'])

    cf_end_time = time.time();
    cf_duration = cf_end_time - cf_start_time;
    sidebar.write("Runtime: " + str(cf_duration) + " s")

elif model_choice == 'Hybrid-Based':
    st.session_state['refresh_btn'] = False
    st.session_state['newly_rated_book_count'] = 0
    hy_start_time = time.time();
    if 'js_hy_recommendations' not in st.session_state:
        st.session_state['js_hy_recommendations'] = hybrid_model.get_hybrid_recommendations(st.session_state['existing_user_id'])
    hy_end_time = time.time();
    hy_duration = hy_end_time - hy_start_time;
    sidebar.write("Runtime: "+ str(hy_duration) + " s")

# TODO : Add clear session state before leaving this page


# Function to display recommendations


def recommended_book_display(recommendations):
    # Create the HTML table to display recommended books
    for index, book in recommendations.iterrows():
        table_html = "<table style='border-collapse: collapse; width: 100%;'>"
        table_html += f"<tr><td style='border: none; width: 22%'><img src='{book['Image-URL-S']}' style='width: 100px; height: 150px;'></td>"
        table_html += f"<td style='border: none;'><strong>Title:</strong> {book['Book-Title']}<br>"
        table_html += f"<strong>ISBN:</strong> {book['ISBN']}<br>"
        table_html += f"<strong>Author:</strong> {book['Book-Author']}<br>"
        table_html += f"<strong>Publisher:</strong> {book['Publisher']}<br>"
        table_html += f"<strong>Year:</strong> {book['Year-Of-Publication']}</td></tr>"
        table_html += "</table>"

        # Display the recommendation table
        st.write(table_html, unsafe_allow_html=True)

        # Display rate button for rating
        col1, col2, col3, col4, col5 = st.columns(5, gap="large")
        with col5:
            rate_button = st.button("Rate", key=f"rate_{index}")
        if rate_button:
            rate_book(book['ISBN'])
        else:
            pass


# Display selected book from select box
def selected_book_display(selected_book):
    book = st.session_state['df_books'][st.session_state['df_books']['Book-Title'] == selected_book].iloc[0]

    # Create the HTML table to display selected books
    table_html = "<table style='border-collapse: collapse; width: 100%;'>"
    table_html += f"<tr><td style='border: none;'><img src='{book['Image-URL-S']}' style='width: 100px;'></td>"
    table_html += f"<td style='border: none;'><strong>Title:</strong> {book['Book-Title']}<br>"
    table_html += f"<strong>ISBN:</strong> {book['ISBN']}<br>"
    table_html += f"<strong>Author:</strong> {book['Book-Author']}<br>"
    table_html += f"<strong>Publisher:</strong> {book['Publisher']}<br>"
    table_html += f"<strong>Year:</strong> {book['Year-Of-Publication']}</td></tr>"
    table_html += "</table>"

    # Display the HTML table
    st.write(table_html, unsafe_allow_html=True)

    # Display rate button for rating
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")
    with col5:
        rate_btn = st.button("Rate")
    if rate_btn:
        rate_book(book['ISBN'])
    else:
        pass


# Function to rate book
def rate_book(isbn):
    rating_container = st.empty()

    # Form to display when rate button is clicked
    with rating_container.form("rate_book", clear_on_submit=True, border=True):
        st.text_input(label="ISBN", disabled=True, value=isbn, key='isbn')  # ISBN get directly from the recommendation table
        st.slider(label="Please rate from 1 to 10: ", min_value=1, max_value=10, step=1, value=None, key='rating')
        st.form_submit_button("Submit", on_click=validate_rating)


# Function to recommendations when refresh button is clicked
def update_recommendations():
    if model_choice == 'Collaborative Filtering-Based':
        query_result = db.collection('ratings').where("user_id", "==", st.session_state['existing_user_id']).get()

        new_data = []
        for doc in query_result:
            new_data.append(doc.to_dict())

        df_new_data = pd.DataFrame(new_data).rename(columns={"user_id": "User-ID",
                                                            "isbn": "ISBN",
                                                            "book_rating": "Book-Rating"})

        st.session_state['yj_cf_recommendations'] = cf_model.update_recommendations(df_new_data, st.session_state['existing_user_id'])
    elif model_choice == 'Content-Based':
        pass
    elif model_choice == 'Hybrid-Based':
        st.session_state['js_hy_recommendations'] = hybrid_model.get_hybrid_recommendations(st.session_state['existing_user_id'])


# Display either recommended book or selected book
if selected_book is None:
    st.markdown("Recommended books:")
    if model_choice == 'Content-Based':
        recommended_book_display(st.session_state['jr_content_recommendations'])
    elif model_choice == 'Collaborative Filtering-Based':
        recommended_book_display(st.session_state['yj_cf_recommendations'])
    elif model_choice == 'Hybrid-Based':
        recommended_book_display(st.session_state['js_hy_recommendations'])
else:
    st.markdown("Selected books:")
    selected_book_display(selected_book)

# Display a refresh button to refresh recommendation list
st.divider()
refresh_btn = st.button("Refresh", disabled=st.session_state['refresh_btn'], on_click=update_recommendations)

