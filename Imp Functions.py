import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import tensorflow as tf
from tensorflow import keras

# Function to preprocess data
def preprocess_data(df, target_column, scale=False, encode=False):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if encode:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train a linear regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to train a logistic regression model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Function to train a random forest classifier
def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model

# Function to train an SVM classifier
def train_svm(X_train, y_train, kernel='linear'):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, regression=False):
    y_pred = model.predict(X_test)
    if regression:
        return mean_squared_error(y_test, y_pred)
    else:
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)

# Function to build a simple neural network
def build_nn(input_shape, output_units, activation='relu'):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=activation, input_shape=(input_shape,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(output_units, activation='softmax' if output_units > 1 else 'sigmoid')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' if output_units > 1 else 'binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train a neural network
def train_neural_network(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
