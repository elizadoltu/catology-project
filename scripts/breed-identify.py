import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nn_training_2 import forward_propagation, initialize_parameters
import nltk
import sys

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data_path = "./data/Cats_database.xlsx"
data = pd.read_excel(data_path)

# Drop irrelevant columns
irrelevant_columns = ['Row.names', 'Timestamp', 'Additional Info']
data_cleaned = data.drop(columns=irrelevant_columns, errors='ignore')

# Extract features and target
X = data_cleaned.drop(columns=['Breed'])
y = data_cleaned['Breed']

# Encode target (Breed)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize neural network parameters
input_size = X_scaled.shape[1]
hidden_layer_size = 10
output_size = len(np.unique(y_encoded))
weights, biases = initialize_parameters(input_size, hidden_layer_size, output_size)

# Define the breed mapping
breed_mapping = {
    1: "Bengal",
    2: "Birman",
    3: "British Shorthair",
    4: "Chartreux",
    5: "European",
    6: "Maine Coon",
    7: "Persian",
    8: "Ragdoll",
    9: "Sphynx",
    10: "Siamese",
    11: "Turkish Angora",
    12: "Other",
    13: "No Breed",
    14: "Unknown",
    15: "Savannah"
}

# Preprocess description
def preprocess_description(description, feature_names):
    tokens = word_tokenize(description.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    
    # Map the description to the dataset features
    feature_vector = {feature: 0 for feature in feature_names}
    for word in filtered_tokens:
        for feature in feature_names:
            if feature.lower() in word:
                feature_vector[feature] += 1
    
    return list(feature_vector.values())

# Predict breed
def predict_breed(description, weights, biases, scaler, feature_names):
    # Preprocess description
    attributes_vector = preprocess_description(description, feature_names)
    scaled_attributes = scaler.transform([attributes_vector])
    
    # Perform forward propagation
    input_data = np.array(scaled_attributes)
    predictions, _ = forward_propagation(input_data, weights, biases)
    
    # Get the predicted breed index
    predicted_index = np.argmax(predictions)
    return predicted_index

# Main function
def main():
    # Get description from command-line input
    if len(sys.argv) < 2:
        print("Usage: python script_name.py 'description of the cat'")
        sys.exit(1)
    
    description = sys.argv[1]  # User inputted description
    feature_names = X.columns.tolist()
    
    # Predict breed
    predicted_index = predict_breed(description, weights, biases, scaler, feature_names)
    predicted_breed = breed_mapping.get(predicted_index + 1, "Unknown Breed")  # Adjust for 1-based mapping
    print(f"Predicted Breed: {predicted_breed}")

if __name__ == "__main__":
    main()
