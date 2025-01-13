import pandas as pd
import random
import numpy as np

# Load the existing database
data_path = "./data/Cats_database.xlsx"
data = pd.read_excel(data_path)

# Drop irrelevant columns
irrelevant_columns = ['Row.names', 'Timestamp', 'Additional Info']
data_cleaned = data.drop(columns=irrelevant_columns, errors='ignore')

# Column-specific value ranges and options
gender_options = [0, 1]  # 0: Female, 1: Male
age_group_options = [1, 2, 3, 4]  # Age groups
breed_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
housing_type_options = [1, 2, 3]
area_options = [1, 2, 3]
outside_access_options = [0, 1, 2, 3]
behavior_traits_range = (1, 5)  # For columns like Shy, Calm, etc.
abundance_of_prey_range = (0, 5)  # Range for prey-related columns

# Function to generate a random row
def generate_random_row():
    row = {
        "Gender": random.choice(gender_options),
        "Age Group": random.choice(age_group_options),
        "Breed": random.choice(breed_options),
        "Number of Cats": random.randint(1, 6),
        "Housing Type": random.choice(housing_type_options),
        "Area": random.choice(area_options),
        "Outside Access": random.choice(outside_access_options),
    }
    # Add behavior traits
    for column in data_cleaned.columns:
        if column not in row.keys():
            if "Predation" in column or "Abundance" in column:
                row[column] = random.randint(*abundance_of_prey_range)
            else:
                row[column] = random.randint(*behavior_traits_range)
    return row

# Generate new rows
num_new_entities = 5000  # Adjust this to add more or fewer entities
new_rows = [generate_random_row() for _ in range(num_new_entities)]

# Convert new rows to a DataFrame
new_data = pd.DataFrame(new_rows)

# Append new data to the original DataFrame
updated_data = pd.concat([data_cleaned, new_data], ignore_index=True)

# Save the updated DataFrame to a new Excel file
output_path = "./data/Updated_Cats_database.xlsx"
updated_data.to_excel(output_path, index=False)

print(f"{num_new_entities} new entities added and saved to {output_path}.")
