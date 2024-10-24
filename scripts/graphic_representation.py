import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data from the provided file path
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print("Datele au fost incarcate cu succes!")
        return data
    except Exception as e:
        print(f"Datele nu au putut fi incarcate. Eroare: {e}")
        return None

# Function to plot histograms and boxplots and save the boxplots in the specified folder
def plot_histogram_boxplot(data, numeric_columns, save_folder='auto/pictures'):
    # Ensure the 'auto/pictures' folder exists, create it if it doesn't
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for column in numeric_columns:
        plt.figure(figsize=(14, 6))

        # Plot the histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data[column], kde=True, bins=20)
        plt.title(f'Histograma pentru {column}')

        # Plot the boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot pentru {column}')

        # Save the boxplot as an image in the 'auto/pictures' folder
        boxplot_path = os.path.join(save_folder, f'boxplot_{column}.png')
        plt.savefig(boxplot_path)  # Save the figure
        print(f"Boxplot pentru {column} salvat la: {boxplot_path}")

        # Close the plot to free memory and avoid displaying it
        plt.close()

# Main function
def main():
    file_path = './data/Cats_database.xlsx'

    data = load_data(file_path)

    if data is not None:
        # Select numeric columns, excluding 'Row.names' and 'Observations'
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_columns = [col for col in numeric_columns if col not in ['Row.names', 'Observations']]
        
        # Plot and save the figures
        plot_histogram_boxplot(data, numeric_columns)

if __name__ == '__main__':
    main()
