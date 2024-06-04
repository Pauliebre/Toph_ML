#Este es el código principal que realiza la clasificación de imagenes de EEG
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Paso 1: Cargar el archivo CSV procesado
df = pd.read_csv('D:/kathy/Downloads/EMFUTECH/ML_CODE/MIRAI_templates/EEG_AURA_RFClassification-main/CSV/csv_fusionado_processed.csv')

# Paso 2: Dividir los datos en características y etiquetas
X = df[['Mean', 'STD', 'Asymmetry']]
y = df['Label']

# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba con más aleatoriedad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Paso 4: Estandarizar las features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Paso 5 Wavelet Transform
import pywt

def wavelet_transform(signals, wavelet='db4', level=4):
    coeffs = []
    for signal in signals:
        coeff = pywt.wavedec(signal, wavelet, level=level)
        coeffs.append(coeff)
    return np.array(coeffs)

# Apply wavelet transform to the training and testing data
X_train_wavelet = wavelet_transform(X_train_scaled)
X_test_wavelet = wavelet_transform(X_test_scaled)

#Build the WNN model

def build_wnn(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define input shape based on wavelet-transformed data
input_shape = X_train_wavelet.shape[1:]

# Build and compile the WNN model
wnn_model = build_wnn(input_shape)

# Train the model
wnn_model.fit(X_train_wavelet, y_train, epochs=50, batch_size=32, validation_data=(X_test_wavelet, y_test))

# Evaluate the model
loss, accuracy = wnn_model.evaluate(X_test_wavelet, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions
y_pred = wnn_model.predict(X_test_wavelet)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()