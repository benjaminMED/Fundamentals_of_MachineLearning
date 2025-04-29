import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Sample dataset creation
np.random.seed(0)
size = 100
X1 = np.random.rand(size) * 10   # Feature 1 (univariate)
X2 = np.random.rand(size) * 5    # Feature 2 (bivariate/multivariate)
X3 = np.random.rand(size) * 8    # Feature 3 (multivariate)
y = 3 * X1 + 2 * X2 + 1.5 * X3 + np.random.randn(size) * 2  # Target

# Creating a DataFrame
df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y
})

# -------------------------------
# 1. Univariate Regression (X1 -> y)
# -------------------------------
print("\n--- Univariate Regression ---")
X_uni = df[['X1']]
y_uni = df['y']

X_train, X_test, y_train, y_test = train_test_split(X_uni, y_uni, test_size=0.2, random_state=0)
model_uni = LinearRegression()
model_uni.fit(X_train, y_train)

y_pred_uni = model_uni.predict(X_test)
print("Coefficient:", model_uni.coef_)
print("Intercept:", model_uni.intercept_)
print("R2 Score:", r2_score(y_test, y_pred_uni))

# Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_uni, color='red', linewidth=2, label='Predicted')
plt.title("Univariate Linear Regression")
plt.xlabel("X1")
plt.ylabel("y")
plt.legend()
plt.show()

# -------------------------------
# 2. Bivariate Regression (X1, X2 -> y)
# -------------------------------
print("\n--- Bivariate Regression ---")
X_bi = df[['X1', 'X2']]
y_bi = df['y']

X_train, X_test, y_train, y_test = train_test_split(X_bi, y_bi, test_size=0.2, random_state=0)
model_bi = LinearRegression()
model_bi.fit(X_train, y_train)

y_pred_bi = model_bi.predict(X_test)
print("Coefficients:", model_bi.coef_)
print("Intercept:", model_bi.intercept_)
print("R2 Score:", r2_score(y_test, y_pred_bi))

# -------------------------------
# 3. Multivariate Regression (X1, X2, X3 -> y)
# -------------------------------
print("\n--- Multivariate Regression ---")
X_multi = df[['X1', 'X2', 'X3']]
y_multi = df['y']

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=0)
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)
print("Coefficients:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)
print("R2 Score:", r2_score(y_test, y_pred_multi))
