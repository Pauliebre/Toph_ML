import pandas as pd

# Step 1: Load the data
df = pd.read_csv('D:\\kathy\\Downloads\\EMFUTECH\\Toph_ML\\dary_mi.csv')
print("Dataframe importado:", df.shape)

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

print("X_completo:", X_completo.shape)
print("X_delta:", X_delta.shape)
print("X_theta:", X_theta.shape)
print("X_alpha:", X_alpha.shape)
print("X_beta:", X_beta.shape)
print("X_gamma:", X_gamma.shape)
print("y:", y.shape)
