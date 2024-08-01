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

# Step 1: Load the data
df = pd.read_csv('C:/Users/edgar/OneDrive/Documentos/MIRAI INNOVATION/EMFUTECH/EMFUTECH_2024_01/NRC/Toph_ML/concatenated_output.csv')

# Reemplazar los valores de la columna "Countdown Type", donde first countdown (0) es reposo, 
# second countdown (1) es flexion izq, third countdown (2) es extension izq, fourth countdown (3) es flex der y fifth(4) es extension der
df['Countdown Type'] = df['Countdown Type'].replace({
    'First Countdown': 0,
    'Second Countdown': 1,
    'Third Countdown': 2,
    'Fourth Countdown': 3,
    'Fifth Countdown': 4
})

# Step 2: Split data into features and labels
X_completo = df[['Cycle', 'Delta 1', 'Delta 2', 'Delta 3', 'Delta 4', 'Delta 5', 'Delta 6', 'Delta 7', 'Delta 8',
                 'Theta 1', 'Theta 2', 'Theta 3', 'Theta 4', 'Theta 5', 'Theta 6', 'Theta 7', 'Theta 8',
                 'Alpha 1', 'Alpha 2', 'Alpha 3', 'Alpha 4', 'Alpha 5', 'Alpha 6', 'Alpha 7', 'Alpha 8',
                 'Beta 1', 'Beta 2', 'Beta 3', 'Beta 4', 'Beta 5', 'Beta 6', 'Beta 7', 'Beta 8',
                 'Gamma 1', 'Gamma 2', 'Gamma 3', 'Gamma 4', 'Gamma 5', 'Gamma 6', 'Gamma 7', 'Gamma 8']]

X_delta = df[['Delta 1', 'Delta 2', 'Delta 3', 'Delta 4', 'Delta 5', 'Delta 6', 'Delta 7', 'Delta 8']]
X_theta = df[['Theta 1', 'Theta 2', 'Theta 3', 'Theta 4', 'Theta 5', 'Theta 6', 'Theta 7', 'Theta 8']]
X_alpha = df[['Alpha 1', 'Alpha 2', 'Alpha 3', 'Alpha 4', 'Alpha 5', 'Alpha 6', 'Alpha 7', 'Alpha 8']]
X_beta = df[['Beta 1', 'Beta 2', 'Beta 3', 'Beta 4', 'Beta 5', 'Beta 6', 'Beta 7', 'Beta 8']]
X_gamma = df[['Gamma 1', 'Gamma 2', 'Gamma 3', 'Gamma 4', 'Gamma 5', 'Gamma 6', 'Gamma 7', 'Gamma 8']]

y = df['Countdown Type']

# Step 3: Split data into training and testing sets with more randomness
X_train, X_test, y_train, y_test = train_test_split(X_completo, y, test_size=0.2, random_state=42, shuffle=True)

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
