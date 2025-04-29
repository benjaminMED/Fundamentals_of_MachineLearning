import numpy as np

# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Perceptron Training Algorithm
def train_perceptron(X, y, lr=0.1, epochs=10):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for i in range(n_samples):
            linear_output = np.dot(X[i], weights) + bias
            y_pred = step_function(linear_output)
            error = y[i] - y_pred
            weights += lr * error * X[i]
            bias += lr * error

    return weights, bias

# Prediction
def predict(X, weights, bias):
    return step_function(np.dot(X, weights) + bias)

# Input features and labels (AND logic gate)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND operation

# Train the model
weights, bias = train_perceptron(X, y)
print("Trained Weights:", weights)
print("Trained Bias:", bias)

# Predict
preds = predict(X, weights, bias)
print("Predictions:", preds)
