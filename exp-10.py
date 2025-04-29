import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data  # Features (30 features about cell nuclei)
y = cancer.target  # Labels (Malignant: 1, Benign: 0)

# Step 2: Apply PCA for dimensionality reduction (reduce to 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 3: Plot the reduced data (2D)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(label='Tumor Type (0: Benign, 1: Malignant)')
plt.title('PCA for Dimensionality Reduction on Breast Cancer Dataset (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Step 4: Explained variance ratio (optional)
print(f"Explained variance ratio for the 2 components: {pca.explained_variance_ratio_}")
