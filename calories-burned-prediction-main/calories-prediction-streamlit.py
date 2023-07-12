import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import plotly.graph_objects as go
import time

from matplotlib import style
style.use("seaborn")
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

st.title("CalBurnIQ: The Intelligent Calorie Burn Calculator")
st.image("https://i.pinimg.com/originals/5c/26/08/5c2608995c3c8e0bf420898a065d082a.gif", width=240)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body.Only thing you have to do is pass your parameters such as `Age` , `Gender` , `BMI` , etc into this WebApp and then you will be able to see the predicted value of kilocalories that burned in your body.")


st.sidebar.header("User Input Parameters : ")

def user_input_features():
    global age , bmi , duration , heart_rate , body_temp
    age = st.sidebar.slider("Age : " , 10 , 100 , 30)
    bmi = st.sidebar.slider("BMI : " , 15 , 40 , 20)
    duration = st.sidebar.slider("Duration (min) : " , 0 , 35 , 15)
    heart_rate = st.sidebar.slider("Heart Rate : " , 60 , 130 , 80)
    body_temp = st.sidebar.slider("Body Temperature (C) : " , 34 , 42 , 38)
    gender_button = st.sidebar.radio("Gender : ", ("Male" , "Female"))

    if gender_button == "Male":
        gender = 1
    else:
        gender = 0

    data = {
    "age" : age,
    "bmi" : bmi,
    "duration" : duration,
    "heart_rate" : heart_rate,
    "body_temp" : body_temp,
    "gender" : ["Male" if gender_button == "Male" else "Female"]
    }

    data_model = {
    "Age" : age,
    "BMI" : bmi,
    "Duration" : duration,
    "Heart_Rate" : heart_rate,
    "Body_Temp" : body_temp,
    "Gender_male" : gender
    }

    features = pd.DataFrame(data_model, index=[0])
    data = pd.DataFrame(data, index=[0])
    return features , data

df , data = user_input_features()

st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)

calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories , on = "User_ID")
# st.write(exercise_df.head())
exercise_df.drop(columns = "User_ID" , inplace = True)

exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.2 , random_state = 1)

for data in [exercise_train_data , exercise_test_data]:         # adding BMI column to both training and test sets
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"] , 2)

exercise_train_data = exercise_train_data[["Gender" , "Age" , "BMI" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_test_data = exercise_test_data[["Gender" , "Age" , "BMI"  , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first = True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first = True)

X_train = exercise_train_data.drop("Calories" , axis = 1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories" , axis = 1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators = 1000 , max_features = 3 , max_depth = 6)
random_reg.fit(X_train , y_train)
random_reg_prediction = random_reg.predict(X_test)

print("RandomForest Mean Absolute Error(MAE) : " , round(metrics.mean_absolute_error(y_test , random_reg_prediction) , 2))
print("RandomForest Mean Squared Error(MSE) : " , round(metrics.mean_squared_error(y_test , random_reg_prediction) , 2))
print("RandomForest Root Mean Squared Error(RMSE) : " , round(np.sqrt(metrics.mean_squared_error(y_test , random_reg_prediction)) , 2))
print("RandomForest R-squared(R2) : " , round(metrics.r2_score(y_test , random_reg_prediction) , 2))
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

st.write(round(prediction[0] , 2) , "   *kilocalories*")

st.write("---")
st.header("Similar Results : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

range = [prediction[0] - 10 , prediction[0] + 10]
ds = exercise_df[(exercise_df["Calories"] >= range[0]) & (exercise_df["Calories"] <= range[-1])]
st.write(ds.sample(5))

st.write("---")
st.header("General Information : ")

boolean_age = (exercise_df["Age"] < age).tolist()
boolean_duration = (exercise_df["Duration"] < duration).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < body_temp).tolist()
boolean_heart_rate= (exercise_df["Heart_Rate"] < heart_rate).tolist()

# bt_age = round(sum(boolean_age) / len(boolean_age) , 2) * 100
# Create the plot
age_percentile = round(sum(boolean_age) / len(boolean_age) , 2) * 100
fig = go.Figure()
fig.add_trace(go.Scatter(x=exercise['Age'], y=exercise['Age'], mode='lines', name='Age'))
fig.add_trace(go.Scatter(x=[age_percentile, age_percentile], y=[0, age], mode='lines', name='Age Percentile'))
st.plotly_chart(fig)

st.write("You are older than %" ,  age_percentile, "of other people.")

# bt_age = round(sum(boolean_age) / len(boolean_age) , 2) * 100
dur_percentile = round(sum(boolean_duration) / len(boolean_duration) , 2) * 100
fig = go.Figure()
fig.add_trace(go.Scatter(x=exercise['Duration'], y=exercise['Duration'], mode='lines', name='Duration'))
fig.add_trace(go.Scatter(x=[dur_percentile, dur_percentile], y=[0, duration], mode='lines', name='Duration Percentile'))
st.plotly_chart(fig)

st.write("Your had higher exercise duration than %" , dur_percentile , "of other people.")

hr_percentile = round(sum(boolean_heart_rate) / len(boolean_heart_rate) , 2) * 100
fig = go.Figure()
fig.add_trace(go.Scatter(x=exercise['Heart_Rate'], y=exercise['Heart_Rate'], mode='lines', name='Heart Rate'))
fig.add_trace(go.Scatter(x=[hr_percentile, hr_percentile], y=[0, heart_rate], mode='lines', name='Heart rate Percentile'))
st.plotly_chart(fig)
st.write("You had more heart rate than %" , hr_percentile , "of other people during exercise.")

# print("Accurary of the model: ", training_data_acc)
# x_train_prediction = model.predict(X_train)

# bt_percentile = round(sum(boolean_body_temp) / len(boolean_body_temp) , 2) * 100
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=exercise['Body_Temp'], y=exercise['Body_Temp'], mode='lines', name='Body temperature'))
# fig.add_trace(go.Scatter(x=[bt_percentile, bt_percentile], y=[0, body_temp], mode='lines', name='Body temperature Percentile'))
# st.plotly_chart(fig)
# st.write("You had higher body temperature  than %" , bt_percentile , "of other people during exercise.")