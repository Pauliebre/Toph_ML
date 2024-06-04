#Este es el código principal que realiza la clasificación de imagenes de EEG con CNN tradicional
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier


from sklearn.model_selection import cross_val_score

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

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

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # For multiclass classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=1, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions
y_pred_prob = model.predict(X_test_scaled)

# Convert predicted probabilities to class labels
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

#Calcular accuracy with cross-validation - Not necessary right now.
#model_cv = KerasClassifier(model, epochs=5, batch_size=10, verbose=0)
#scores = cross_val_score(model_cv, X_train_scaled, y_train, cv=5)  # 5-fold cross-validation

# Print the accuracy for each fold
#print("Cross-validation scores: ", scores)
#print("Mean accuracy: ", np.mean(scores))
#print("Standard deviation: ", np.std(scores))
