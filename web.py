"""This module contains necessary functions needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import streamlit as st

@st.cache_data
def load_data():
    """This function returns the preprocessed data"""
    # Load the Heart Disease dataset into DataFrame.
    df = pd.read_csv('heart.csv')
    # Perform feature and target split
    X = df[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]]
    y = df['HeartDisease']
    return df, X, y

@st.cache_data
def train_model(X, y):
    """This function trains the model and returns the model and model score"""
    # Ensure X and y are writable numpy arrays
    X = np.array(X)
    y = np.array(y)
    # Create the model
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)
    # Return the values
    return model, score

def predict(X, y, features):
    """This function makes predictions using the trained model"""
    # Get model and model score
    model, score = train_model(X, y)
    # Ensure features is a writable numpy array
    features = np.array(features).astype(np.float64)
    # Predict the value
    prediction = model.predict(features.reshape(1, -1))
    return prediction, score

def build_lstm(num_units, input_shape):
    """This function builds an LSTM model"""
    # Define input shape
    input_layer = tf.keras.layers.Input(shape=input_shape)
    # LSTM layer
    lstm_layer = tf.keras.layers.LSTM(num_units, return_sequences=True)(input_layer)
    # Output layer
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_layer)
    # Define model
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
    return model
