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
        sns.boxplot(data[column])
        plt.title(f'Boxplot pentru {column}')
        plt.tight_layout()

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle('Boxplot-uri pentru toate atributele', fontsize=16)
    plt.show()