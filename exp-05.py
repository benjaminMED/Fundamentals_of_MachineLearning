import numpy as np

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  # XOR output

# Seed for reproducibility
np.random.seed(1)

# Initialize weights
input_layer_size = 2
hidden_layer_size = 4
output_layer_size = 1

# Random weights and biases
W1 = np.random.rand(input_layer_size, hidden_layer_size)
b1 = np.zeros((1, hidden_layer_size))
W2 = np.random.rand(hidden_layer_size, output_layer_size)
b2 = np.zeros((1, output_layer_size))

# Training loop
epochs = 10000
lr = 0.1

for _ in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Backpropagation
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Prediction
pred = final_output.round()
print("Predictions:\n", pred)
