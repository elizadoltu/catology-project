import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Define loss functions
def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Prevent log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred.T), axis=1))

def accuracy(y_true, y_pred):
    y_pred_labels = np.argmax(y_pred, axis=0)
    y_true_labels = np.argmax(y_true, axis=1)
    return np.mean(y_true_labels == y_pred_labels)

# Initialize parameters
def initialize_parameters(input_size, hidden_layer_size, output_size):
    np.random.seed(42)
    weights = {
        "W1": np.random.uniform(-0.05, 0.05, (hidden_layer_size, input_size)),
        "W2": np.random.uniform(-0.05, 0.05, (output_size, hidden_layer_size)),
    }
    biases = {
        "b1": np.zeros((hidden_layer_size, 1)),
        "b2": np.zeros((output_size, 1)),
    }
    return weights, biases

# Forward propagation
def forward_propagation(X, weights, biases):
    Z1 = np.dot(weights["W1"], X.T) + biases["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(weights["W2"], A1) + biases["b2"]
    A2 = softmax(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Backward propagation
def backward_propagation(X, y, weights, biases, cache, learning_rate):
    m = X.shape[0]
    A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]
    delta_k = A2 - y.T
    delta_h = A1 * (1 - A1) * np.dot(weights["W2"].T, delta_k)

    dW2 = np.dot(delta_k, A1.T)
    dW1 = np.dot(delta_h, X)
    db2 = np.sum(delta_k, axis=1, keepdims=True)
    db1 = np.sum(delta_h, axis=1, keepdims=True)

    weights["W1"] -= learning_rate * dW1 / m
    weights["W2"] -= learning_rate * dW2 / m
    biases["b1"] -= learning_rate * db1 / m
    biases["b2"] -= learning_rate * db2 / m

    return weights, biases

# Train the network
def train_network(X_train, y_train, weights, biases, learning_rate, epochs):
    losses = []
    for epoch in range(epochs):
        A2, cache = forward_propagation(X_train, weights, biases)
        loss = cross_entropy_loss(y_train, A2)
        losses.append(loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
        weights, biases = backward_propagation(X_train, y_train, weights, biases, cache, learning_rate)
    return weights, biases, losses

# Main function
def main():
    data_path = "./data/Cats_database.xlsx"
    data = pd.read_excel(data_path)
    data_cleaned = data.drop(columns=['Row.names', 'Timestamp', 'Additional Info'], errors='ignore')
    X = data_cleaned.drop(columns=['Breed'])
    y = data_cleaned['Breed']

    # Encode target
    one_hot_encoder = OneHotEncoder()
    y_encoded = one_hot_encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Initialize parameters
    input_size = X_train.shape[1]
    hidden_layer_size = 10
    output_size = y_encoded.shape[1]
    learning_rate = 0.01
    epochs = 500
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
    A2_test, _ = forward_propagation(X_test, weights, biases)
    test_loss = cross_entropy_loss(y_test, A2_test)
    test_accuracy = accuracy(y_test, A2_test) * 100
    print(f"Final Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
