# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load and Prepare Data
def load_and_prepare_data(file_path):
    data = pd.read_excel(file_path)
    data_cleaned = data.drop(columns=['Row.names', 'Timestamp', 'Additional Info'], errors='ignore')
   # Separate features and target
    X = data_cleaned.drop(columns=['Breed'])  # Remove 'Breed' from features
    y = data_cleaned['Breed']  # Set 'Breed' as the target variable

    scaler = MinMaxScaler() #de verificat si cu si fara asta
    X_normalized = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42) #here we actually split the data 80/20
    return X_train, X_test, y_train, y_test

# Step 2: Initialize Parameters
def initialize_parameters(input_size, hidden_layer_size, output_size): #1 strat ascuns si crestem progresiv
    np.random.seed(42)
    weights = {
        "W1": np.random.randn(hidden_layer_size, input_size) * 0.01,
        "W2": np.random.randn(output_size, hidden_layer_size) * 0.01,
    }
    biases = {
        "b1": np.zeros((hidden_layer_size, 1)),
        "b2": np.zeros((output_size, 1)),
    }
    return weights, biases

# Step 3: Define Activation and Loss Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred) / len(y_true)

# # Step 4: Forward Propagation
# def forward_propagation(X, weights, biases):
#     Z1 = np.dot(weights["W1"], X.T) + biases["b1"]
#     A1 = sigmoid(Z1)
#     Z2 = np.dot(weights["W2"], A1) + biases["b2"]
#     A2 = sigmoid(Z2)
#     return Z1, A1, Z2, A2

# # Step 5: Backward Propagation
# def backward_propagation(X, y, Z1, A1, Z2, A2, weights):
#     m = X.shape[0]
#     y = y.values.reshape(1, -1)
#     dZ2 = A2 - y
#     dW2 = (1 / m) * np.dot(dZ2, A1.T)
#     db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
#     dA1 = np.dot(weights["W2"].T, dZ2)
#     dZ1 = dA1 * sigmoid_derivative(Z1)
#     dW1 = (1 / m) * np.dot(dZ1, X)
#     db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
#     gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
#     return gradients

# # Step 6: Train the Network
# def train(X_train, y_train, weights, biases, learning_rate, max_epochs):
#     for epoch in range(max_epochs):
#         Z1, A1, Z2, A2 = forward_propagation(X_train, weights, biases)
#         loss = mse_loss(y_train, A2.flatten())
#         gradients = backward_propagation(X_train, y_train, Z1, A1, Z2, A2, weights)
#         weights["W1"] -= learning_rate * gradients["dW1"]
#         biases["b1"] -= learning_rate * gradients["db1"]
#         weights["W2"] -= learning_rate * gradients["dW2"]
#         biases["b2"] -= learning_rate * gradients["db2"]
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {loss:.4f}")
#     return weights, biases

# # Step 7: Evaluate the Model
# def evaluate(X_test, y_test, weights, biases):
#     _, _, _, A2 = forward_propagation(X_test, weights, biases)
#     predictions = A2.flatten()
#     mse = mse_loss(y_test, predictions)
#     print(f"Test MSE: {mse:.4f}")
#     return predictions

# # Main function to execute the steps
# if __name__ == "__main__":
#     file_path = './data/Cats_database.xlsx'
#     X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)
#     input_size = X_train.shape[1]
#     hidden_layer_size = 10
#     output_size = 1
#     learning_rate = 0.01
#     max_epochs = 1000
#     weights, biases = initialize_parameters(input_size, hidden_layer_size, output_size)
#     weights, biases = train(X_train, y_train, weights, biases, learning_rate, max_epochs)
#     predictions = evaluate(X_test, y_test, weights, biases)
