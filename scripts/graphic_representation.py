import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def loat_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print("Datele au fost incarcate cu succes!")
        return data
    except Exception as e:
        print(f"Datele nu au putut fi incarcate. Eroare: {e}")
        return None
    
def plot_histogram_boxplot(data, numeric_columns):
    for column in numeric_columns:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(data[column], kde=True, bins=20)
        plt.title(f'Histograma pentru {column}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot pentru {column}')

        plt.tight_layout()
        plt.show()

def main():
    file_path = './data/Translated_Data_cat_personality_and_predation.xlsx'

    data = loat_data(file_path)

    if data is not None:
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        plot_histogram_boxplot(data, numeric_columns)

if __name__ == '__main__':
    main()