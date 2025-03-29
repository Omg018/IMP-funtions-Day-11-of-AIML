# AI/ML Functions in Python

This repository provides a collection of AI and Machine Learning functions implemented in Python using popular libraries like TensorFlow, Scikit-Learn, NumPy, and Pandas. The script is designed to handle data preprocessing, model training, evaluation, and prediction.

## 📌 Features

### 🔹 Data Preprocessing
- **Scaling and Encoding**: Standardizes numerical data and encodes categorical labels for better model performance.
- **Train-Test Split**: Automatically splits datasets into training and testing sets.

### 🔹 Supervised Learning Models
- **Linear Regression**: Used for predicting continuous values based on input features.
- **Logistic Regression**: A classification model used for binary/multi-class classification tasks.
- **Random Forest Classifier**: An ensemble learning model that improves classification performance.
- **Support Vector Machine (SVM)**: A powerful classification model using different kernel functions.

### 🔹 Model Evaluation
- **Accuracy Score**: Measures the correctness of classification models.
- **Classification Report**: Provides precision, recall, and F1-score for classification tasks.
- **Mean Squared Error (MSE)**: Evaluates the performance of regression models.

### 🔹 Neural Network (Deep Learning)
- **Simple ANN (Artificial Neural Network)**: Built using TensorFlow/Keras with fully connected layers.
- **Customizable Activation Functions**: Supports ReLU, Softmax, and Sigmoid activations.

## 📥 Installation

To use this script, install the required dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow
```

## 🚀 Usage

Import the functions from the script and apply them to your AI/ML projects.

### 🔹 Example Usage

#### 1️⃣ Preprocessing the Data

```python
import pandas as pd
from aiml_functions import preprocess_data

df = pd.read_csv('data.csv')
X_train, X_test, y_train, y_test = preprocess_data(df, target_column='target', scale=True, encode=True)
```

#### 2️⃣ Training a Logistic Regression Model

```python
from aiml_functions import train_logistic_regression

model = train_logistic_regression(X_train, y_train)
```

#### 3️⃣ Evaluating the Model

```python
from aiml_functions import evaluate_model

accuracy, report = evaluate_model(model, X_test, y_test)
print(f'Accuracy: {accuracy}')
print(report)
```

#### 4️⃣ Training a Neural Network

```python
from aiml_functions import build_nn, train_neural_network

nn_model = build_nn(input_shape=X_train.shape[1], output_units=2)
nn_model = train_neural_network(nn_model, X_train, y_train, epochs=10, batch_size=32)
```

## 🤝 Contributing

Contributions are welcome! Feel free to submit pull requests to enhance functionality, improve performance, or add new AI/ML features.
