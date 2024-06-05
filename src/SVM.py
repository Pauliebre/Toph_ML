import pandas as pd
import numpy as np
import pywt
import itertools
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the processed CSV file
df = pd.read_csv('D:/kathy/Downloads/EMFUTECH/ML_CODE/MIRAI_templates/EEG_AURA_RFClassification-main/CSV/csv_fusionado_processed.csv')

# Step 2: Split data into features and labels
X = df[['Mean', 'STD', 'Asymmetry']]
y = df['Label']

# Step 3: Split data into training and testing sets with more randomness
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Step 4: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to NumPy array
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Define a function to perform wavelet transformation
def wavelet_transform(data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = np.concatenate(coeffs, axis=-1)
    return coeffs

# Apply wavelet transformation to the features
X_train_wavelet = np.array([wavelet_transform(sample) for sample in X_train_scaled])
X_test_wavelet = np.array([wavelet_transform(sample) for sample in X_test_scaled])

# Reshape the data if necessary
X_train_wavelet = X_train_wavelet.reshape(X_train_wavelet.shape[0], -1)
X_test_wavelet = X_test_wavelet.reshape(X_test_wavelet.shape[0], -1)

# Define the SVM model
svm_model = SVC()

# Define the hyperparameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_wavelet, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f'Best Accuracy: {grid_search.best_score_:.4f}')
print(f'Best Hyperparameters: {best_params}')

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_wavelet)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))
