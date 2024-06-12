import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

# Step 1: Load the data
df = pd.read_csv('D:\kathy\Downloads\EMFUTECH\Toph_ML\datos_test\MI\dary_mi.csv')

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

# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba con más aleatoriedad
X_train, X_test, y_train, y_test = train_test_split(X_completo, y, test_size=0.2, random_state=42, shuffle=True)

# Paso 4: Entrenar el clasificador Random Forest
clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Paso 5: Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Paso 6: Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {accuracy}")
# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()

with open('random_forest.pkl', 'wb') as f:
    pickle.dump(clf, f)