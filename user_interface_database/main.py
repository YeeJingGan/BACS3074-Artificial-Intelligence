import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd

# Establish connection to database
try:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\ai_assignment\firebase\ai-assignment-2885a-firebase-adminsdk-ls4zo-1155c285b4.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

# Function to check if user_id exits in the database
df_existing_users = pd.read_csv(r'csvs/Users.csv') # Users from CSV
new_users_in_database = db.collection('users').stream()
df_new_users = pd.DataFrame([doc.to_dict() for doc in new_users_in_database]) # Users from Database


def check_valid_user(user_id):
    if user_id in df_existing_users['User-ID'].values or user_id in df_new_users['user_id'].values:
        return True
    else:
        return False


# Session state
if 'existing_user_id' not in st.session_state:
    st.session_state['existing_user_id'] = 0

if 'user_id_created' not in st.session_state:
    st.session_state['user_id_created'] = False

# Create login form
login_page = st.empty()
with login_page.form("login", clear_on_submit=True, border=False):
    st.markdown("## Welcome to BookStore")
    st.session_state['existing_user_id'] = st.number_input("User ID", min_value=0)
    submit_btn = st.form_submit_button("Login")

    # Check if user_id valid
    is_user = check_valid_user(st.session_state['existing_user_id'])
    if submit_btn and is_user:
        st.switch_page("pages/home.py")
    elif submit_btn and is_user is False:
        st.error("Invalid user id!")
    else:
        pass

# Direct to sign up
st.markdown("")
st.page_link("pages/sign_up.py", label=f":blue[Not a user? Sign up Now!]", icon="✍", disabled=st.session_state['user_id_created'])


# Show created user ID
if 'user_id' in st.session_state and st.session_state['user_id'] != 0:
    success_message = "Your user ID is " + str(st.session_state['user_id'])
    st.success(success_message)