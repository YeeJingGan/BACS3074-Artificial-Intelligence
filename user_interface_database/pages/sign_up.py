import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Establish connection to database
try:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\ai_assignment\firebase\ai-assignment-2885a-firebase-adminsdk-ls4zo-1155c285b4.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

# Function to generate new user id


def generate_new_user_id(age, location):
    latest_users = db.collection('users').order_by('user_id', direction=firestore.Query.DESCENDING).limit(1).get()
    latest_user_document_id = latest_users[0].reference.id
    latest_user_id = latest_users[0].get("user_id")
    db.collection('users').document(str(int(latest_user_document_id) + 1)).set({'user_id': latest_user_id + 1,
                                                                                'age': age,
                                                                                'location': location
                                                                                })
    return latest_user_id + 1


# Session state and call back
if 'generate_btn_disabled' not in st.session_state:
    st.session_state.generate_btn_disabled = False

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = 0

def call_back(age, location):
    st.session_state.is_disable = True if age and location.strip() != '' else False
    st.session_state['user_id_created'] = True if age and location.strip() != '' else False


# Main display
st.markdown("## Not a user?")
st.markdown("Provide your age and location to generate a user id")
age = st.number_input("Age: ", min_value=1, max_value=100)
location = st.text_input("Location: ")

if st.button("Generate", disabled= st.session_state.is_disable if 'is_disable' in st.session_state else False, on_click= call_back(age,location)):
    if age is None or location.strip() == '':
        st.error("Age and location is required field")
    else:
        st.session_state['user_id'] = generate_new_user_id(age, location)
        st.switch_page("main.py")
