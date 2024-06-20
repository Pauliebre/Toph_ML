import pandas as pd
import tensorflow as tf
import numpy as np
import pywt
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Step 1: Load the data
df = pd.read_csv('D:\kathy\Downloads\EMFUTECH\Toph_ML\datos_test\MI\concatenated_output.csv')

# Reemplazar los valores de la columna "Countdown Type"
df['Countdown Type'] = df['Countdown Type'].replace({
    'First Countdown': 0,
    'Second Countdown': 1,
    'Third Countdown': 2,
    'Fourth Countdown': 3,
    'Fifth Countdown': 4
})

# Step 2: Split data into features and labels
X_completo = df[['Delta 1', 'Delta 2', 'Delta 3', 'Delta 4', 'Delta 5', 'Delta 6', 'Delta 7', 'Delta 8',
                 'Theta 1', 'Theta 2', 'Theta 3', 'Theta 4', 'Theta 5', 'Theta 6', 'Theta 7', 'Theta 8',
                 'Alpha 1', 'Alpha 2', 'Alpha 3', 'Alpha 4', 'Alpha 5', 'Alpha 6', 'Alpha 7', 'Alpha 8',
                 'Beta 1', 'Beta 2', 'Beta 3', 'Beta 4', 'Beta 5', 'Beta 6', 'Beta 7', 'Beta 8',
                 'Gamma 1', 'Gamma 2', 'Gamma 3', 'Gamma 4', 'Gamma 5', 'Gamma 6', 'Gamma 7', 'Gamma 8']]
y = df['Countdown Type']

print (f"X_completo shape:{X_completo.shape}")

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_completo, y, test_size=0.2, random_state=42, shuffle=True)
print (f"X_train shape:{X_train.shape}")


# Step 4: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print (f"X_scaled shape:{X_train_scaled.shape}")

# Save the fitted scaler
scaler_file_path = 'scaler.joblib'
joblib.dump(scaler, scaler_file_path)
print(f'Scaler saved to {scaler_file_path}')

# Convert labels to NumPy array
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Define a function to perform wavelet transformation
def wavelet_transform(data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = np.concatenate(coeffs, axis=-1)
    return coeffs

# Define the function to create the Keras model
def create_model(input_dim, optimizer='adam', init_mode='normal', activation='relu', neurons=128):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(neurons // 2, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(5, kernel_initializer=init_mode, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the hyperparameter grid for wavelet transform
wavelet_grid = {
    'wavelet': ['db4', 'coif1', 'sym5'],
    'level': [1, 3, 5]
}

# Generate all combinations of wavelet hyperparameters
all_wavelet_combinations = list(itertools.product(wavelet_grid['wavelet'], wavelet_grid['level']))

best_accuracy = 0
best_wavelet_params = None

# Perform wavelet transform hyperparameter search
for wavelet, level in all_wavelet_combinations:
    X_train_wavelet = np.array([wavelet_transform(sample, wavelet=wavelet, level=level) for sample in X_train_scaled])
    X_test_wavelet = np.array([wavelet_transform(sample, wavelet=wavelet, level=level) for sample in X_test_scaled])
    
    input_dim = X_train_wavelet.shape[1]
    print (f"input dimension shape = Xtrain_wavelet shape:{input_dim}")

    # Create the model with fixed hyperparameters
    model = create_model(input_dim=input_dim)
    
    # Train the model using cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, val_idx in kfold.split(X_train_wavelet):
        X_train_cv, X_val_cv = X_train_wavelet[train_idx], X_train_wavelet[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        history = model.fit(X_train_cv, y_train_cv, epochs=10, batch_size=30, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val_cv, y_val_cv, verbose=0)
        accuracies.append(val_accuracy)
    
    mean_accuracy = np.mean(accuracies)
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_wavelet_params = (wavelet, level)

print(f"BIEN IMPORTANTE SHAPE AFTER WAVELET during search: {X_train_wavelet.shape}")
print(f"BIEN IMPORTANTE input dimension during search: {input_dim}")
print(f'Best Accuracy: {best_accuracy:.4f}')
print(f'Best Wavelet Parameters: wavelet={best_wavelet_params[0]}, level={best_wavelet_params[1]}')

# Train the best model on the full training set with best wavelet transform parameters
wavelet, level = best_wavelet_params
X_train_wavelet = np.array([wavelet_transform(sample, wavelet=wavelet, level=level) for sample in X_train_scaled])
X_test_wavelet = np.array([wavelet_transform(sample, wavelet=wavelet, level=level) for sample in X_test_scaled])

print(f"x train WAVELET usado en entrenamiento: {X_train_wavelet.shape}")

best_model = create_model(input_dim=X_train_wavelet.shape[1])
best_model.fit(X_train_wavelet, y_train, epochs=20, batch_size=20, validation_split=0.2, verbose=1)

print(f"!!!! best model created using: {X_train_wavelet.shape[1]}")

# Evaluate the best model on the test set
loss, accuracy = best_model.evaluate(X_test_wavelet, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions with the best model
y_pred_prob = best_model.predict(X_test_wavelet)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# Save the trained model
model_file_path = 'trained_model.joblib'
joblib.dump(best_model, model_file_path)
print(f'Model saved to {model_file_path}')