import pandas as pd
import glob
import os 
# Define the pattern to match your CSV files
file_pattern = "*final.csv"

# Find all matching CSV files
file_paths = glob.glob(file_pattern)

# Create an empty list to store dataframes
all_dfs = []

# Loop through each file, read into a dataframe, and append to the list
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)
        df['source_file'] = os.path.basename(file_path)
        all_dfs.append(df)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file (optional)
combined_df.to_csv("combined_data.csv", index=False) 
