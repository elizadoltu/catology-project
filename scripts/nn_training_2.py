import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler  
import matplotlib.pyplot as plt 

# Step 1: Define activation and loss functions

def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # Sigmoid activation function

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))  # Derivative of sigmoid

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred.flatten()) ** 2)  # Mean Squared Error (MSE)

def accuracy(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).astype(int)  # Convert predictions to binary (0 or 1)
    return np.mean(y_true == y_pred_labels)  # Calculate accuracy

# Step 2: Initialize parameters (weights and biases)

def initialize_parameters(input_size, hidden_layer_size, output_size):
    np.random.seed(42)  # Set a random seed for reproducibility
    weights = {
        "W1": np.random.uniform(-0.05, 0.05, (hidden_layer_size, input_size)),  # Initialize W1
        "W2": np.random.uniform(-0.05, 0.05, (output_size, hidden_layer_size)),  # Initialize W2
    }
    biases = {
        "b1": np.zeros((hidden_layer_size, 1)),  # Initialize biases for hidden layer
        "b2": np.zeros((output_size, 1)),  # Initialize biases for output layer
    }
    return weights, biases

# Step 3: Forward propagation

def forward_propagation(X, weights, biases):
    Z1 = np.dot(weights["W1"], X.T) + biases["b1"]  # Linear transformation for hidden layer
    A1 = sigmoid(Z1)  # Apply sigmoid activation
    Z2 = np.dot(weights["W2"], A1) + biases["b2"]  # Linear transformation for output layer
    A2 = sigmoid(Z2)  # Apply sigmoid activation
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}  # Store intermediate values
    return A2, cache


def backward_propagation(X, y, weights, biases, cache, learning_rate):
    m = X.shape[0]  # Number of training examples
    A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]  # Retrieve cached values

    # Error term for output layer (T4.3) , adica diferenta intre outputul prezis A2 si outputul real y
    delta_k = A2 - y.T  # Output error term

    # Error term for hidden layer (T4.4), adica cat de mult influenteaza fiecare neuron din hidden layer la output
    delta_h = A1 * (1 - A1) * np.dot(weights["W2"].T, delta_k)

    # Gradients for weights and biases
    dW2 = np.dot(delta_k, A1.T) # dW2 = delta_k * A1, gradient for W2
    dW1 = np.dot(delta_h, X)# dW1 = delta_h * X, gradient for W1
    db2 = np.sum(delta_k, axis=1, keepdims=True) # db2 = delta_k, gradient for b2
    db1 = np.sum(delta_h, axis=1, keepdims=True) # db1 = delta_h, gradient for b1

    # Update weights and biases (T4.5)
    weights["W1"] -= learning_rate * dW1 / m # the weights are updated with the gradients multiplied by the learning rate and divided by the number of training examples
    weights["W2"] -= learning_rate * dW2 / m 
    biases["b1"] -= learning_rate * db1 / m # the biases are updated with the gradients multiplied by the learning rate and divided by the number of training examples
    biases["b2"] -= learning_rate * db2 / m

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
    data_path = "./data/Cats_database.xlsx"
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

    # Calculate and print training accuracy
    A2_train, _ = forward_propagation(X_train, weights, biases)
    train_accuracy = accuracy(y_train, A2_train) * 100
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    # Calculate and print test accuracy
    test_accuracy = accuracy(y_test, predictions) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
