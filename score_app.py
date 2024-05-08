import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the dataset
dataset = pd.read_csv('ipl.csv')

# Extracting input features and labels
X = dataset.iloc[:, [7, 8, 9, 12, 13]].values  # Input features
y = dataset.iloc[:, 14].values  # Label

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function to predict score
def predict_score(model, input_data):
    return model.predict(sc.transform(np.array([input_data])))[0]

# Streamlit app
st.title('Cricket Score Prediction')

# Input features
runs = st.number_input('Runs', value=100)
wickets = st.number_input('Wickets', value=0)
overs = st.number_input('Overs', value=13)
striker = st.number_input('Striker', value=50)
non_striker = st.number_input('Non-Striker', value=50)

# Choose model
model_choice = st.radio('Choose Model', ('Linear Regression', 'Random Forest Regression'))

if st.button('Predict'):
    if model_choice == 'Linear Regression':
        lin_regressor = LinearRegression()
        lin_regressor.fit(X_train, y_train)
        prediction = predict_score(lin_regressor, [runs, wickets, overs, striker, non_striker])
    elif model_choice == 'Random Forest Regression':
        rf_regressor = RandomForestRegressor(n_estimators=100, max_features=None, random_state=0)
        rf_regressor.fit(X_train, y_train)
        prediction = predict_score(rf_regressor, [runs, wickets, overs, striker, non_striker])

    st.success(f'Predicted Score: {prediction}')
