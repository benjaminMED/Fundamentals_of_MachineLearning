import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

n = len(x)

# Calculating slope (m) and intercept (c)
m = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
c = (np.sum(y) - m * np.sum(x)) / n

# Prediction
y_pred = m * x + c

print(f"Slope (m): {m:.2f}")
print(f"Intercept (c): {c:.2f}")

# Plot
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='LSM Line')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression using LSM")
plt.legend()
plt.show()
