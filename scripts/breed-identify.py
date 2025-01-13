import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.preprocessing import MinMaxScaler
import pickle
import gensim.downloader as api

# Import the forward_propagation function from your MLP file
# If your training code is in "nn_training_with_embeddings.py", import from there.
# Or if it's in "nn_training_2.py", do:
from nn_training_2 import forward_propagation  # or adapt as needed

# Make sure these downloads are done
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

############################################
# 1. LOAD PRETRAINED WORD EMBEDDING MODEL
############################################
print("Loading GloVe model. This may take a moment...")
glove_model = api.load("glove-wiki-gigaword-50")
print("GloVe model loaded successfully!")

####################################
# 2. LOAD TRAINED WEIGHTS & SCALER
####################################
# We retrieve the same weights, biases, and scaler from the training step
# so the inference matches the training environment exactly.
with open("model_weights.pkl", "rb") as f:
    weights = pickle.load(f)

with open("model_biases.pkl", "rb") as f:
    biases = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

###################################
# 3. LOAD TRAINING DATA FOR COLUMNS
###################################
# We do this to replicate how we built numeric_df in training.
# That way we know how many columns (k) were used.
data_path = "./data/Updated_Cats_database.xlsx"
data = pd.read_excel(data_path)
irrelevant_columns = ['Row.names', 'Timestamp']
data_cleaned = data.drop(columns=irrelevant_columns, errors='ignore')

# Suppose 'Observations' is the text column we singled out in training
text_column = 'Observations'
numeric_df = data_cleaned.drop(columns=['Breed', text_column], errors='ignore')

# The columns we used as numeric features
feature_names = numeric_df.columns.tolist()

###############################################
# 4. SYNONYM + EMBEDDING UTILS (SAME AS TRAIN)
###############################################
def expand_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def get_text_embedding(tokens):
    vectors = []
    for token in tokens:
        if token in glove_model:
            vectors.append(glove_model[token])
    if not vectors:
        return np.zeros(glove_model.vector_size)
    return np.mean(vectors, axis=0)

##############################################
# 5. PROCESS USER DESCRIPTION INTO EMBEDDING
##############################################
def process_text(description):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(str(description).lower())
    filtered_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]
    
    expanded_tokens = []
    for token in filtered_tokens:
        expanded_tokens.append(token)
        syns = expand_synonyms(token)
        expanded_tokens.extend(list(syns)[:2])
    
    embedding = get_text_embedding(expanded_tokens)  # shape (50,)
    return embedding

##############################################
# 6. BUILD (1, k+50) VECTOR FOR INFERENCE
##############################################
def build_input_vector(description):
    """
    1) Create a numeric "keyword" vector of length k, using the same columns 
       as in training.
    2) Create a 50-dim text embedding from the user's description.
    3) Concatenate => (1, k+50).
    4) Use the *same* scaler to transform it, ensuring shapes match.
    """
    # Step 1) Build a dictionary for the numeric columns (k) if relevant
    # This logic can vary; for example, you might count occurrences 
    # or you might do more advanced feature engineering.
    feature_dict = {col: 0 for col in feature_names}
    
    # Example: if user typed a token that matches a numeric col name, increment it.
    # (You can adapt this to your real logic from training if needed.)
    user_tokens = word_tokenize(description.lower())
    for token in user_tokens:
        for col in feature_names:
            if col.lower() in token:  # simplistic match
                feature_dict[col] += 1
    
    numeric_part = pd.DataFrame([feature_dict])[feature_names].to_numpy()  # shape: (1, k)
    
    # Step 2) 50-d embedding
    text_emb = process_text(description)  # shape = (50,)
    text_emb = text_emb.reshape(1, 50)
    
    # Step 3) Concatenate => shape = (1, k+50)
    combined = np.hstack((numeric_part, text_emb))
    
    # Step 4) Scale => shape = (1, k+50)
    scaled = scaler.transform(combined)
    return scaled

#############################################
# 7. FORWARD PASS & PREDICTION
#############################################
def predict_breed(description):
    input_data = build_input_vector(description)  # shape (1, k+50)
    predictions, _ = forward_propagation(input_data, weights, biases)
    print("Prediction Probabilities:", predictions.flatten())
    predicted_index = np.argmax(predictions)
    return predicted_index

#################################
# 8. MAIN INTERACTION
#################################
def main():
    description = input("Enter cat description: ")
    predicted_index = predict_breed(description)
    
    # Example breed mapping; adapt if your index->breed is different
    breed_mapping = {
        1: "Bengal", 2: "Birman", 3: "British Shorthair", 4: "Chartreux", 
        5: "European", 6: "Maine Coon", 7: "Persian", 8: "Ragdoll",
        9: "Sphynx", 10: "Siamese", 11: "Turkish Angora", 12: "Other", 
        13: "No Breed", 14: "Unknown", 15: "Savannah"
    }
    predicted_breed = breed_mapping.get(predicted_index + 1, "Unknown Breed")
    print(f"Predicted Breed: {predicted_breed}")

if __name__ == "__main__":
    main()
