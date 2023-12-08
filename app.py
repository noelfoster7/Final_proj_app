import streamlit as st
import pandas as pd
import numpy as np


file_path = '/Users/noelfoster/Desktop/stream_folder/social_media_usage.csv'
ss = pd.read_csv(file_path)

def clean_sm(x):
    return np.where(x == 1, 1, 0)
#Make sm_li variable
s['sm_li'] = clean_sm(s['web1h'])

#Select columns
selected_columns = ['income', 'educ2', 'par', 'marital', 'gender', 'age', 'sm_li']

#Drop na's
ss = s[selected_columns].dropna()

#Creat x and y
y = ss['sm_li']
X = ss.drop(columns=['sm_li'])

from sklearn.model_selection import train_test_split

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')

# Fit model w/ training data
model.fit(X_train, y_train)



st.markdown("# This is my Final Project")

st.markdown("Noel Foster 12/06/2023")

user_name = st.text_input("What's your name?")

if user_name:
    st.write(f"Hello {user_name}, welcome to my app!")

st.write("### Gender")
gender = st.selectbox("What is your gender?",
        options = ["Male",
                    "Female",
                    "Other",
                    "Don't know",
                    "Prefer not to answer"])

if gender == "Male":
    gender = 1
elif gender == "Female":
    gender = 2
elif gender == "Other":
    gender = 3
elif gender == "Don't know":
    gender = 98
else:
    gender = 99
st.write(f"Gender number number: {gender}")

st.write("### Age")
Age = st.slider(label="What's your age?",
min_value=1,
max_value=97,
value=1)
st.write(f"Your age: {Age}")

st.write("### Relationship Status")
mar = st.selectbox("What is your relationship status?",
        options = ["Married",
                    "Living with a partner",
                    "Divorced",
                    "Seperated",
                    "Widowed",
                    "Never been married"])

if mar == "Married":
    mar = 1
elif mar == "Living with a partner":
    mar = 2
elif mar == "Divorced":
    mar = 3
elif mar == "Seperated":
    mar = 4
elif mar == "Widowed":
    mar = 5
else:
    mar = 6
st.write(f"Relationship status number: {mar}")

st.write("### Education")
educ2 = st.selectbox("What is your furtherst education?",
        options = ["Less than high school",
                    "High school incomplete",
                    "High School Graduate",
                    "Some college, no degree",
                    "Two-year associates degree",
                    "Four-year college or university/Bachelors degree",
                    "Some postgraduate schooling, no postgrad degree",
                    "Postgrad or professional degree"])

if educ2 == "Less than high school":
    educ2 = 1
elif educ2 == "High school incomplete":
    educ2 = 2
elif educ2 == "High School Graduate":
    educ2 = 3
elif educ2 == "Some college, no degree":
    educ2 = 4
elif educ2 == "Two-year associates degree":
    educ2 = 5
elif educ2 == "Four-year college or university/Bachelors degree":
    educ2 = 6
elif educ2 == "Some postgraduate schooling, no postgrad degree":
    educ2 = 7
else:
    educ2 = 8
st.write(f"Furthest education number: {educ2}")

st.write("### Income")
income_mapping = {
    '<10,000': 1,
    '10,000-19,999': 2,
    '20,000-29,999': 3,
    '30,000-39,999': 4,
    '40,000-49,999': 5,
    '50,000-74,999': 6,
    '75,000-99,999': 7,
    '100,000-149,999': 8,
    '150,000+': 9
}

selected_income = st.select_slider(label='Select Income Range:',
                                   options=list(income_mapping.keys()))

# Display the corresponding mapped value
if selected_income:
    mapped_value = income_mapping[selected_income]
    st.write(f"Income number: {mapped_value}")



st.write("### Parental Status")
par = st.selectbox("Are you a parent?",
        options = ["Yes",
                    "No"])

if par == "Yes":
    par = 1
elif par == "No":
    par = 2
else:
    par = 3
st.write(f"Parent number: {par}")

st.write("### LinkedIn Status")
LinkedIn = st.selectbox("Do you use LinkedIn?",
        options = ["Yes",
                    "No", "I don't know"])

if LinkedIn == "Yes":
    LinkedIn = 1
elif LinkedIn == "No":
    LinkedIn = 2
else:
    LinkedIn = 3
st.write(f"LinkedIn Number: {LinkedIn}")

selected_income_numeric = income_mapping.get(selected_income)

x = np.array([selected_income_numeric, educ2, par, mar, gender, Age]).reshape(1,-1)
prob_x = model.predict_proba(x)[:, 1]
st.write("### This is the probability you use LinkedIn")
st.write(f"Probability of being a LinkedIn user: {prob_x[0]}")
