import pandas as pd
import os

def concatenate_csv_files_from_folder(folder_path):
    # Create an empty list to store individual DataFrames
    dataframes = []
    
    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    return concatenated_df

# Leer files de carpeta de MI
folder_path = 'D:\kathy\Downloads\EMFUTECH\Toph_ML\datos_test\MI'  # Replace with the path to your folder containing CSV files
concatenated_df = concatenate_csv_files_from_folder(folder_path)

# Save the concatenated DataFrame to a new CSV file
output_path = 'D:\kathy\Downloads\EMFUTECH\Toph_ML\datos_test\MI/concatenated_output.csv'  # Replace with your desired output path
concatenated_df.to_csv(output_path, index=False)

print(f"Concatenated CSV saved to {output_path}")
