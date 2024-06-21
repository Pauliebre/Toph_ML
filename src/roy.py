#This code detects the ERD/ERS ratio in trial data, afterwards it compares and makes data analysis between data acquisitions previous to motor imagery and posterior to motor imagery training.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
data = pd.read_csv("D:\kathy\Downloads\EMFUTECH\Toph_ML\datos_test\MI\dary_mi.csv")

# Filter out 'First Countdown' in 'Countdown Type'
filtered_data = data[data['Countdown Type'] != 'First Countdown']

# Select only alpha (1-8) and beta (1-8) columns
alpha_columns = [f'Alpha {i}' for i in range(1, 9)]
beta_columns = [f'Beta {i}' for i in range(1, 9)]
selected_columns = alpha_columns + beta_columns + ['Countdown Type', 'Cycle']

filtered_data = filtered_data[selected_columns]

# Function to drop the first and last 10 rows of each block of data
def drop_first_last_10_rows(df):
    if len(df) > 20:
        return df.iloc[10:-10]
    else:
        return pd.DataFrame(columns=df.columns)

# Map the countdown type labels to new names
countdown_label_map = {
    'Second Countdown': 'LA Flexion',
    'Third Countdown': 'LA Extension',
    'Fourth Countdown': 'RA Flexion',
    'Fifth Countdown': 'RA Extension'
}
filtered_data['Countdown Type'] = filtered_data['Countdown Type'].map(countdown_label_map)

# Initialize dictionaries to store maximum changes for each countdown type and cycle
max_changes = {'Alpha': {}, 'Beta': {}}
change_directions = {'Alpha': {}, 'Beta': {}}

# Store rows into different dataframes based on 'Countdown Type' and 'Cycle'
countdown_types = filtered_data['Countdown Type'].unique()
cycle_values = filtered_data['Cycle'].unique()

for countdown_type in countdown_types:
    max_changes['Alpha'][countdown_type] = []
    max_changes['Beta'][countdown_type] = []
    change_directions['Alpha'][countdown_type] = []
    change_directions['Beta'][countdown_type] = []
    for cycle in cycle_values:
        cycle_data = filtered_data[(filtered_data['Countdown Type'] == countdown_type) & (filtered_data['Cycle'] == cycle)]
        if not cycle_data.empty:
            cleaned_data = drop_first_last_10_rows(cycle_data)
            if not cleaned_data.empty:
                # Calculate row-wise averages for alpha and beta columns
                cleaned_data['Average Alpha'] = cleaned_data[alpha_columns].mean(axis=1)
                cleaned_data['Average Beta'] = cleaned_data[beta_columns].mean(axis=1)
                
                # Calculate the difference between consecutive rows
                cleaned_data['Alpha Diff'] = cleaned_data['Average Alpha'].diff()
                cleaned_data['Beta Diff'] = cleaned_data['Average Beta'].diff()
                
                # Find the maximum amplitude change and its direction
                max_alpha_change = cleaned_data['Alpha Diff'].abs().max()
                max_alpha_change_direction = 'ERS' if cleaned_data['Alpha Diff'].max() == max_alpha_change else 'ERD'
                
                max_beta_change = cleaned_data['Beta Diff'].abs().max()
                max_beta_change_direction = 'ERS' if cleaned_data['Beta Diff'].max() == max_beta_change else 'ERD'
                
                max_changes['Alpha'][countdown_type].append(max_alpha_change)
                max_changes['Beta'][countdown_type].append(max_beta_change)
                change_directions['Alpha'][countdown_type].append(max_alpha_change_direction)
                change_directions['Beta'][countdown_type].append(max_beta_change_direction)

# Convert max changes to DataFrame for easier plotting
max_changes_df = pd.DataFrame({
    'RM activity': np.repeat(countdown_types, len(cycle_values)),
    'Cycle': np.tile(cycle_values, len(countdown_types)),
    'Max Mu Change': [max_changes['Alpha'][ct][i] for ct in countdown_types for i in range(len(cycle_values))],
    'Polarization Mu': [change_directions['Alpha'][ct][i] for ct in countdown_types for i in range(len(cycle_values))],
    'Max Beta Change': [max_changes['Beta'][ct][i] for ct in countdown_types for i in range(len(cycle_values))],
    'Polarization Beta': [change_directions['Beta'][ct][i] for ct in countdown_types for i in range(len(cycle_values))]
})

# 1) Graph of the maximum mu change for each different countdown type
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=max_changes_df, x='RM activity', y='Max Mu Change', hue='Polarization Mu')
plt.title('Maximum Mu Change for Each RM activity')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.show()

# 2) Graph of the maximum beta change for each different countdown type
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=max_changes_df, x='RM activity', y='Max Beta Change', hue='Polarization Beta')
plt.title('Maximum Beta Change for Each RM activity')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.show()

# 3) Graph of the Average of all countdown types, average maximum mu change
avg_max_mu_change = max_changes_df.groupby('RM activity')['Max Mu Change'].mean().mean()
mu_change_direction = 'ERS' if max_changes_df.groupby('RM activity')['Max Mu Change'].mean().max() == avg_max_mu_change else 'ERD'
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=['Average Max Mu Change'], y=[avg_max_mu_change])
plt.title('Average of All RM activities, Average Maximum Mu Change')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f} ({mu_change_direction})', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.show()

# 4) Graph of the Average of all countdown types, average maximum beta change
avg_max_beta_change = max_changes_df.groupby('RM activity')['Max Beta Change'].mean().mean()
beta_change_direction = 'ERS' if max_changes_df.groupby('RM activity')['Max Beta Change'].mean().max() == avg_max_beta_change else 'ERD'
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=['Average Max Beta Change'], y=[avg_max_beta_change])
plt.title('Average of All RM activities, Average Maximum Beta Change')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f} ({beta_change_direction})', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.show()

# 5) Graph of the average of each countdown type maximum mu change depending on the cycle
plt.figure(figsize=(10, 6))
sns.lineplot(data=max_changes_df, x='Cycle', y='Max Mu Change', hue='RM activity', marker='o', style='Polarization Mu')
plt.title('Average Maximum Mu Change by Cycle for Each RM activity')
plt.show()

# 6) Graph of the average of each countdown type maximum beta change depending on the cycle
plt.figure(figsize=(10, 6))
sns.lineplot(data=max_changes_df, x='Cycle', y='Max Beta Change', hue='RM activity', marker='o', style='Polarization Beta')
plt.title('Average Maximum Beta Change by Cycle for Each RM activity')
plt.show()

# 7) Graph comparison between mu change average vs beta change average change for each countdown
avg_changes_df = max_changes_df.groupby('RM activity').mean().reset_index()
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=avg_changes_df, x='RM activity', y='Max Mu Change', color='blue', label='Mu')
ax = sns.barplot(data=avg_changes_df, x='RM activity', y='Max Beta Change', color='red', label='Beta')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.legend()
plt.title('Comparison Between Mu Change Average vs Beta Change Average for Each RM activity')
plt.show()
