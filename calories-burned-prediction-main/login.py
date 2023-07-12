import streamlit as st
from pyrebase import initialize_app
import subprocess

# initialize firebase app
config = {
  'apiKey': "AIzaSyD_hcV6wp0sfM2UGVtKvGIa-9lGTq46cb8",
  'authDomain': "hackproject-e1603.firebaseapp.com",
  'projectId': "hackproject-e1603",
  'databaseURL': "https://hackproject-e1603-default-rtdb.firebaseio.com",
  'storageBucket': "hackproject-e1603.appspot.com",
  'messagingSenderId': "214966453980",
  'appId': "1:214966453980:web:6fb191665c810b2fe6d619",
  'measurementId': "G-SWHBS78YDP"
}

firebase = initialize_app(config)
auth = firebase.auth()

def login(email, password):
    try:
        # Verify the password and the email
        user = auth.sign_in_with_email_and_password(email, password)
        return user['localId']
    except Exception as e:
        print(e)
        return None

def login_page():
    st.set_page_config(page_title="Login", page_icon=":guardsman:", layout="wide")
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    if st.button('Submit'):
        user_id = login(email, password)
        if user_id:
            st.success(f"Welcome, {user_id}")
            subprocess.run(["streamlit", "run", "calories-prediction-streamlit.py"])
            return True
        else:
            st.error("Invalid email or password.")
    return False

if not login_page():
    st.warning("Please login to continue.")

