import pandas as pd

def check_dataset(file_path):
    try:
        # Read the dataset from an Excel file
        data = pd.read_excel(file_path)
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            print("Missing values detected:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found.")

        # Check for duplicate instances
        duplicate_instances = data.duplicated().sum()
        if duplicate_instances > 0:
            print(f"Number of duplicate instances detected: {duplicate_instances}")
            print(data[data.duplicated(keep=False)])  # Display all duplicate instances
        else:
            print("No duplicate instances found.")

       # num_columns = data.shape[1]
       # if num_columns != expected_num_columns:  
          #  print(f"Expected number of columns: {expected_num_columns}, found: {num_columns}")
        
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


 


check_dataset('./Cats_database.xlsx')
