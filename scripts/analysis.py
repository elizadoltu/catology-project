import pandas as pd
import os

file_path = './data/Cats_database.xlsx'
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Drop the specified columns
df = df.drop(columns=['Row.names', 'Timestamp', 'Additional Info'], errors='ignore')

output_directory = 'auto'

os.makedirs(output_directory, exist_ok=True)

output_file = os.path.join(output_directory, 'cats_data_analysis.txt')

def display_class_instances(df, file):
    breed_counts = df['Breed'].value_counts()
    file.write("Number of instances for each class (Breed):\n")
    file.write(str(breed_counts))
    file.write("\n\n")
    
def display_distinct_values(df, file):
    for column in df.columns:
        file.write(f"Attribute: {column}\n")
        file.write("Distinct values: " + str(df[column].unique()) + "\n")
        file.write("Global value counts:\n")
        file.write(str(df[column].value_counts()) + "\n")
        file.write("\nValue counts by Breed:\n")
        file.write(str(df.groupby('Breed')[column].value_counts()) + "\n")
        file.write("\n" + "-" * 50 + "\n\n")
        
def display_correlations(df, file):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate the correlation matrix
    correlations = numeric_df.corr()

    # Check if 'Breed' is in the numeric DataFrame
    if 'Breed' in correlations.columns:
        # Get the correlation of all attributes with 'Breed'
        breed_correlations = correlations['Breed'].drop('Breed')

        # Sort attributes based on their absolute correlation with 'Breed'
        sorted_attributes = breed_correlations.abs().sort_values(ascending=False).index
        
        # Create a sorted correlation matrix based on the sorted attributes
        sorted_correlations = correlations.loc[sorted_attributes, sorted_attributes]
        
        file.write("Correlation Matrix (ordered by influence on 'Breed'):\n")
        file.write(str(sorted_correlations))
        file.write("\n\n")
    else:
        file.write("No numeric attributes found to correlate with 'Breed'.\n\n")
    
with open(output_file, 'w', encoding='utf-8') as file:
    display_class_instances(df, file)
    display_distinct_values(df, file)
    display_correlations(df, file)

print(f"Analysis written to {output_file}")
