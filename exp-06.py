# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import numpy as np

# Step 1: Load the LFW (Labeled Faces in the Wild) dataset
lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.data  # The pixel values of the images
y = lfw_people.target  # The labels (the person corresponding to the image)
target_names = lfw_people.target_names  # The names of the people in the dataset

print(f"Loaded {X.shape[0]} images of {len(target_names)} people")

# Step 2: Preprocess the data - Reduce dimensionality using PCA (Principal Component Analysis)
n_components = 150  # Number of components to keep
pca = PCA(n_components=n_components, whiten=True).fit(X)
X_pca = pca.transform(X)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=42)

# Step 4: Train the SVM classifier
model = SVC(kernel='linear', class_weight='balanced')  # SVM with linear kernel
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

