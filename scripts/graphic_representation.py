'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def transform_non_numeric(data):
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le
    return data

def plot_distributions(data):
    data.hist(figsize=(14, 10), bins=20, edgecolor='black')
    plt.suptitle('Histograme pentru toate atributele', fontsize=16)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()

    plt.figure(figsize=(14, 10))
    for i, column in enumerate(data.columns, 1):
        plt.subplot(5, 6, i)
        sns.boxplot(y=data[column])
        plt.title(f'Boxplot pentru {column}')
        plt.tight_layout()

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle('Boxplot-uri pentru toate atributele', fontsize=16)
    plt.show()

file_path = './data/Cats_database.xlsx'
df = pd.read_excel(file_path)

df, laber_encoders = transform_non_numeric(df)
plot_distributions(df)
'''

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