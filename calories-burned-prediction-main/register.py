import streamlit as st
import subprocess
import pyrebase

# initialize firebase app
firebaseConfig = {
  'apiKey': "AIzaSyD_hcV6wp0sfM2UGVtKvGIa-9lGTq46cb8",
  'authDomain': "hackproject-e1603.firebaseapp.com",
  'projectId': "hackproject-e1603",
  'databaseURL': "https://hackproject-e1603-default-rtdb.firebaseio.com",
  'storageBucket': "hackproject-e1603.appspot.com",
  'messagingSenderId': "214966453980",
  'appId': "1:214966453980:web:6fb191665c810b2fe6d619",
  'measurementId': "G-SWHBS78YDP"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()



db = firebase.database()
storage = firebase.storage()    



st.set_page_config(page_title="Register", page_icon=":guardsman:", layout="wide")

st.title("Register")

handle = st.text_input('Please enter your username')
email = st.text_input('Please enter your email address')
password = st.text_input('Please enter your password', type= 'password')
submit = st.button('Sign Up')

if submit:
    user = auth.create_user_with_email_and_password(email,password)
    db.child(user['localId']).child("Handle").set(handle)
    db.child(user['localId']).child("ID").set(user['localId'])
    subprocess.run(["streamlit", "run", "calories-prediction-streamlit.py"])

