import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import tensorflow as tf
import numpy as np
import pywt
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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

# Reshape the data for LSTM input
X_train_wavelet = X_train_wavelet.reshape((X_train_wavelet.shape[0], 1, X_train_wavelet.shape[1]))
X_test_wavelet = X_test_wavelet.reshape((X_test_wavelet.shape[0], 1, X_test_wavelet.shape[1]))

# Define the function to create the LSTM model
def create_model(optimizer='adam', activation='relu', neurons=64):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train_wavelet.shape[1], X_train_wavelet.shape[2]), activation=activation))
    model.add(Dense(neurons // 2, activation=activation))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the hyperparameter grid
param_grid = {
    'batch_size': [10, 20, 30],
    'epochs': [5, 10, 20],
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'neurons': [32, 64, 128]
}

# Generate all combinations of hyperparameters
all_combinations = list(itertools.product(param_grid['batch_size'], param_grid['epochs'], param_grid['optimizer'], param_grid['activation'], param_grid['neurons']))

# Perform random search
n_combinations_to_try = 10  # Number of combinations to try randomly
np.random.seed(42)
random_combinations = np.random.choice(len(all_combinations), n_combinations_to_try, replace=False)

best_accuracy = 0
best_params = None

for idx in random_combinations:
    batch_size, epochs, optimizer, activation, neurons = all_combinations[idx]
    
    # Create the model with current hyperparameters
    model = create_model(optimizer=optimizer, activation=activation, neurons=neurons)
    
    # Train the model using cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, val_idx in kfold.split(X_train_wavelet):
        X_train_cv, X_val_cv = X_train_wavelet[train_idx], X_train_wavelet[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        history = model.fit(X_train_cv, y_train_cv, epochs=epochs, batch_size=batch_size, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val_cv, y_val_cv, verbose=0)
        accuracies.append(val_accuracy)
    
    mean_accuracy = np.mean(accuracies)
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_params = (batch_size, epochs, optimizer, activation, neurons)

print(f'Best Accuracy: {best_accuracy:.4f}')
print(f'Best Hyperparameters: batch_size={best_params[0]}, epochs={best_params[1]}, optimizer={best_params[2]}, activation={best_params[3]}, neurons={best_params[4]}')

# Train the best model on the full training set
batch_size, epochs, optimizer, activation, neurons = best_params
best_model = create_model(optimizer=optimizer, activation=activation, neurons=neurons)
best_model.fit(X_train_wavelet, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

# Evaluate the best model on the test set
loss, accuracy = best_model.evaluate(X_test_wavelet, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions with the best model
y_pred_prob = best_model.predict(X_test_wavelet)
y_pred = np.argmax(y_pred_prob, axis=1)

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
