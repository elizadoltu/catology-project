# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Define activation and loss functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred.flatten()) ** 2)

# Step 2: Initialize parameters
def initialize_parameters(input_size, hidden_layer_size, output_size):
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

# Step 3: Forward propagation
def forward_propagation(X, weights, biases):
    Z1 = np.dot(weights["W1"], X.T) + biases["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(weights["W2"], A1) + biases["b2"]
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Step 4: Backward propagation
def backward_propagation(X, y, weights, biases, cache, learning_rate):
    m = X.shape[0]
    A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]
    dZ2 = A2 - y.T
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(weights["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    weights["W1"] -= learning_rate * dW1
    weights["W2"] -= learning_rate * dW2
    biases["b1"] -= learning_rate * db1
    biases["b2"] -= learning_rate * db2
    return weights, biases

# Step 5: Train the network
def train_network(X_train, y_train, weights, biases, learning_rate, epochs):
    losses = []
    for epoch in range(epochs):
        A2, cache = forward_propagation(X_train, weights, biases)
        loss = mse_loss(y_train, A2.T)
        losses.append(loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
        weights, biases = backward_propagation(X_train, y_train, weights, biases, cache, learning_rate)
    return weights, biases, losses

# Step 6: Evaluate the network
def evaluate_network(X_test, y_test, weights, biases):
    A2_test, _ = forward_propagation(X_test, weights, biases)
    test_loss = mse_loss(y_test, A2_test.T)
    return test_loss, A2_test

# Main execution
def main():
    # Load data
    data_path = r"D:\Facultate\artifIntel\catology-project\data\Cats_database.xlsx"
    data = pd.read_excel(data_path)
    data_cleaned = data.drop(columns=['Row.names', 'Timestamp', 'Additional Info'], errors='ignore')
    X = data_cleaned.drop(columns=['Breed'])
    y = data_cleaned['Breed']

    # Preprocess data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Initialize parameters
    input_size = X_train.shape[1]
    hidden_layer_size = 10
    output_size = 1
    learning_rate = 0.01
    epochs = 200
    weights, biases = initialize_parameters(input_size, hidden_layer_size, output_size)

    # Train the network
    weights, biases, losses = train_network(X_train, y_train, weights, biases, learning_rate, epochs)

    # Plot training loss
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    # Evaluate the network
    test_loss, predictions = evaluate_network(X_test, y_test, weights, biases)
    print(f"Final Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
