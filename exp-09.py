from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Digits dataset
digits = load_digits()
X = digits.data  # Features (images)
y = digits.target  # Labels (digits 0-9)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Step 4: Train the model
knn.fit(X_train, y_train)

# Step 5: Predict using the trained model
y_pred = knn.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
