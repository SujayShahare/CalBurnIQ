import streamlit as st
import subprocess

st.set_page_config(page_title="CalBurnIQ: The Intelligent Calorie Burn Calculator", page_icon=":guardsman:", layout="wide")

st.title("CalBurnIQ: The Intelligent Calorie Burn Calculator")
img_file = "calories2.png"
st.image(img_file, width=300)
st.write("### Welcome to CalBurnIQ app! ")
st.write("#### Our app is designed to help you predict the number of calories you burn during exercise and physical activity.")
st.write("#### With advanced algorithms that take into account your, `weight`, `height`, `BMI` and other factors, our app can provide accurate calorie burn estimates.")
st.write("#### By utilizing CalBurnIQ, you can effectively track progress towards your fitness goals and improve your overall well-being.")


if st.button("Go to Prediction Page"):
    subprocess.run(["streamlit", "run", "calories-prediction-streamlit.py"])
if st.button("Login"):
    subprocess.run(["streamlit", "run", "login.py"])
if st.button("Register"):
    subprocess.run(["streamlit", "run", "register.py"])



