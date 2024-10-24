import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = './data/Translated_Data_cat_personality_and_predation.xlsx'
df = pd.read_excel(file_path)

# Define mappings for the categorical variables
gender_mapping = {'M': 0, 'F': 1,'NSP':3}
age_mapping = {'Moinsde1': 1, '1a2': 2, '2a10': 3, 'Plusde10': 4}
abundance_mapping = {'NSP': 0, '3':3, '2':2, '1':1, '0':0}
# Updated breed mapping using the values you provided
breed_mapping = {
    'BEN': 1, 'SBI': 2, 'BRI': 3, 'CHA': 4, 'EUR': 5, 'MCO': 6, 'PER': 7, 
    'RAG': 8, 'SPH': 9, 'ORI': 10, 'TUV': 11, 'Autre': 12, 'NSP': 13, 'NR':14, 'SAV':15
}

housing_mapping = {'ASB': 1, 'AAB': 2, 'ML': 3, 'MI': 4}
area_mapping = {'U': 1, 'PU': 2, 'R': 3}  # Urban, Periurban, Rural
number_of_cats_mapping = {'1': 1, '2': 2, '3': 3, '4': 4,'5':5, 'Plusde5': 6}

# Apply the mappings to the relevant columns
df['Gender'] = df['Gender'].map(gender_mapping)
df['Age Group'] = df['Age Group'].map(age_mapping)
df['Breed'] = df['Breed'].map(breed_mapping)
df['Housing Type'] = df['Housing Type'].map(housing_mapping)
df['Area'] = df['Area'].map(area_mapping)  # Apply the area mapping
df['Number of Cats'] = df['Number of Cats'].map(number_of_cats_mapping)
df['Abundance of Prey'] = df['Abundance of Prey'].map(abundance_mapping)

# Save the modified DataFrame to a new Excel file
df.to_excel('./data/Cats_database.xlsx', index=False)

# Optional: Create boxplots
# plt.figure(figsize=(14, 10))

# plt.subplot(2, 2, 1)
# sns.boxplot(x=df['Gender'])
# plt.title('Gender Boxplot')

# plt.subplot(2, 2, 2)
# sns.boxplot(x=df['Age Group'])
# plt.title('Age Group Boxplot')

# plt.subplot(2, 2, 3)
# sns.boxplot(x=df['Breed'])
# plt.title('Breed Boxplot')

# plt.subplot(2, 2, 4)
# sns.boxplot(x=df['Housing Type'])
# plt.title('Housing Type Boxplot')

# plt.tight_layout()
# plt.savefig('boxplots_cats.png')

# plt.show()
