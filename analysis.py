import pandas as pd

file_path = 'Cats_database.xlsx'
df = pd.read_excel(file_path, sheet_name="Sheet1")

output_file = 'cats_data_analysis.txt'

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
    correlations = numeric_df.corr()
    file.write("Correlation Matrix:\n")
    file.write(str(correlations))
    file.write("\n\n")
    
with open(output_file, 'w', encoding='utf-8') as file:
    display_class_instances(df, file)
    display_distinct_values(df, file)
    display_correlations(df, file)

print(f"Analysis written to {output_file}")
