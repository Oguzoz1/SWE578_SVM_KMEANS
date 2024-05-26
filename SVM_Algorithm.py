import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib
import os
from enum import Enum

class SVMType(Enum):
    LINEAR = 1
    RBF = 2
    POLY = 3

# Function to check if a model exists and load it
def load_or_train_model(model_name, train_func):
    if os.path.exists(model_name):
        print(f"Loading model from {model_name}")
        model = joblib.load(model_name)
    else:
        print(f"No existing model found. Training new model and saving to {model_name}")
        model = train_func()
        joblib.dump(model, model_name)
    return model

############### DATA PREPARATION ######################
mnist = datasets.fetch_openml('mnist_784', version=1)
# X as Image, y as a label
X, y = mnist["data"], mnist["target"]

X = X / 255.0 # Normalization


# Filtering 2,3,8, and 9
filtered = np.isin(y, ['2', '3', '8', '9'])
X_filtered = X[filtered]
y_filtered = y[filtered]

# Splitting training and test sets. Change random state value to randomize the split.
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=20)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

############### TRAINING  ###############
def train_linear_svc():
    linear = LinearSVC(dual=False, max_iter=5000)
    regularization_parameters = {'C': [0.1, 1, 10]}
    cross_validation_grid_search = GridSearchCV(linear, regularization_parameters, cv=5, n_jobs=4)
    cross_validation_grid_search.fit(X_train, y_train)
    best_linear = cross_validation_grid_search.best_estimator_
    return best_linear

def train_rbf_svc():
    rbf_svc = SVC(kernel='rbf')
    param_grid = {'C': [0.1, 1], 'gamma': [0.001, 0.01, 0.1]}
    grid_search = GridSearchCV(rbf_svc, param_grid, cv=5, n_jobs=4)
    grid_search.fit(X_train_pca, y_train)
    best_rbf_svc = grid_search.best_estimator_
    return best_rbf_svc

############### CHECK AND LOAD/OR TRAIN MODELS ######################

# Linear SVC
linear_model_name = 'linear_svc_model.pkl'
best_linear = load_or_train_model(linear_model_name, train_linear_svc)

# Test for LINEAR
y_pred_test = best_linear.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy (Linear): ", test_accuracy)

# NON-LINEAR RBF 
rbf_model_name = 'rbf_svc_model.pkl'
best_rbf_svc = load_or_train_model(rbf_model_name, train_rbf_svc)

# Test for NON-LINEAR
y_pred_test_rbf = best_rbf_svc.predict(X_test_pca)
test_accuracy_rbf = accuracy_score(y_test, y_pred_test_rbf)
print("Test Accuracy (RBF SVM): ", test_accuracy_rbf)

#Chat GPT mark
############### ANALYZING SUPPORT VECTORS (For RBF) ###############
# Get the support vectors indices in the PCA-transformed data
support_indices = best_rbf_svc.support_

# Map the support vector indices from the PCA-transformed data back to the original data
# Get the support vectors in the PCA-transformed space
support_vectors_pca = X_train_pca[support_indices]

# Get the corresponding original images for these support vectors
support_vectors_original = pca.inverse_transform(support_vectors_pca)

# Convert the support vectors back to their original 28x28 shape
support_vectors_images = support_vectors_original.reshape(-1, 28, 28)

# Labels corresponding to support vectors
support_labels = y_train.iloc[support_indices].values

# Plotting support vectors for digits 2, 3, 8, and 9
def plot_support_vectors(images, labels, support_indices, num_images=5):
    fig, axes = plt.subplots(nrows=4, ncols=num_images, figsize=(10, 8))
    fig.suptitle('Support Vectors for Digits 2, 3, 8, and 9', fontsize=16)

    for i, digit in enumerate(['2', '3', '8', '9']):
        digit_indices = [idx for idx in range(len(support_labels)) if support_labels[idx] == digit]
        if len(digit_indices) >= num_images:
            selected_indices = np.random.choice(digit_indices, num_images, replace=False)
        else:
            selected_indices = digit_indices

        for j, idx in enumerate(selected_indices):
            axes[i, j].imshow(images[idx], cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(digit, fontsize=14)

    plt.show()

# Plot the support vectors
plot_support_vectors(support_vectors_images, support_labels, support_indices)