import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.preprocessing import MinMaxScaler
import pickle
import gensim.downloader as api

# We assume you have a forward_propagation function from nn_training_2 or nn_training_with_embeddings
from nn_training_2 import forward_propagation  # or adapt if your file name differs

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

###########################################
# 1. LOAD PRETRAINED WORD EMBEDDING MODEL
###########################################
print("Loading GloVe model. This may take a moment...")
glove_model = api.load("glove-wiki-gigaword-50")
print("GloVe model loaded successfully!")

####################################
# 2. LOAD TRAINED WEIGHTS & SCALER
####################################
with open("model_weights.pkl", "rb") as f:
    weights = pickle.load(f)

with open("model_biases.pkl", "rb") as f:
    biases = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

###################################
# 3. LOAD TRAINING DATA FOR COLUMNS
###################################
# We'll replicate the numeric columns (k) we used in training
data_path = "./data/Updated_Cats_database.xlsx"
data = pd.read_excel(data_path)
irrelevant_columns = ['Row.names', 'Timestamp']
data_cleaned = data.drop(columns=irrelevant_columns, errors='ignore')

text_column = 'Observations'  # or the name you used
numeric_df = data_cleaned.drop(columns=['Breed', text_column], errors='ignore')
feature_names = numeric_df.columns.tolist()  # The same columns used in training

###################################
# 4. HELPER FUNCTIONS (NLP + EMBED)
###################################
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

def build_input_vector(description):
    """
    Build a single (1, k+50) vector for the classification:
     1) numeric features (k)
     2) 50-dim GloVe embedding
    Then scale it with the same scaler from training.
    """
    # Very simplistic numeric approach: check if column name is in text
    feature_dict = {col: 0 for col in feature_names}
    
    user_tokens = word_tokenize(description.lower())
    for token in user_tokens:
        for col in feature_names:
            if col.lower() in token:
                feature_dict[col] += 1
    
    numeric_part = pd.DataFrame([feature_dict])[feature_names].to_numpy()  # shape (1, k)
    text_emb = process_text(description).reshape(1, 50)                   # shape (1, 50)
    combined = np.hstack((numeric_part, text_emb))                        # shape (1, k+50)
    
    scaled = scaler.transform(combined)                                   # shape (1, k+50)
    return scaled

def predict_breed(description):
    """
    1) Build input vector of shape (1, k+50)
    2) forward_propagation
    3) Argmax for predicted index
    """
    input_data = build_input_vector(description)
    predictions, _ = forward_propagation(input_data, weights, biases)
    predicted_index = np.argmax(predictions)
    return predicted_index, predictions.flatten()

###################################
# 5. BREED DESCRIPTIONS & MAPPING
###################################
# You can store short descriptions for each breed in Romanian.
# Extend or edit as you prefer.
breed_descriptions = {
    1: "Bengal: Pisici extrem de energice, cu blană asemănătoare leopardului, foarte jucăușe.",
    2: "Birman: Recunoscute pentru blana lor mătăsoasă, atitudine blândă și ochii albaștri.",
    3: "British Shorthair: Pisici robuste, independente, cu blană deasă și personalitate calmă.",
    4: "Chartreux: Au o blană cenușie distinctivă, sunt afectuoase și tăcute.",
    5: "European: Denumire generică pentru multe pisici domestice europene, temperament variat.",
    6: "Maine Coon: Printre cele mai mari rase, cu blană lungă și personalitate prietenoasă.",
    7: "Persian: Pisici cu blană lungă, foarte calde și afectuoase, preferă liniștea.",
    8: "Ragdoll: Pisici mari și docile, adesea căutând atenție și purtându-se ca niște 'păpuși de cârpă'.",
    9: "Sphynx: Pisici fără blană, sociabile și inteligente, cer multă atenție.",
    10: "Siamese: Vocalizează frecvent, alerte, extrem de atașate de proprietari, blană cu puncte distincte.",
    11: "Turkish Angora: Subțiri și elegante, foarte jucăușe, cu blană mătăsoasă și adesea albă.",
    12: "Other: O categorie pentru pisici care nu se încadrează în restul raselor.",
    13: "No Breed: Pisici fără rasă distinctă sau mixtă, pot avea diverse caractere.",
    14: "Unknown: Rasă necunoscută, clasificată astfel dacă datele sunt insuficiente.",
    15: "Savannah: Rase exotice, rezultatul încrucișării cu pisici sălbatice (Serval), personalitate energică."
}

breed_mapping = {
    1: "Bengal", 2: "Birman", 3: "British Shorthair", 4: "Chartreux", 
    5: "European", 6: "Maine Coon", 7: "Persian", 8: "Ragdoll",
    9: "Sphynx", 10: "Siamese", 11: "Turkish Angora", 12: "Other", 
    13: "No Breed", 14: "Unknown", 15: "Savannah"
}

#####################################
# 6. GENERATE NATURAL-LANGUAGE TEXT
#####################################
def generate_single_breed_description(description):
    """
    1. Use the trained classifier to predict the breed from the user description.
    2. Return a natural-language statement in Romanian describing that breed.
    """
    pred_index, _ = predict_breed(description)
    # Our dictionary uses keys 1..15
    breed_desc = breed_descriptions.get(pred_index + 1, 
                                        "Nu există descriere disponibilă pentru această rasă.")
    return f"Pe baza descrierii introduse, rasa prezisă este: {breed_mapping.get(pred_index + 1, 'Necunoscut')}\n{breed_desc}"

def generate_comparison(description1, description2):
    """
    1. Predict the breed from the first description.
    2. Predict the breed from the second description.
    3. Return a natural-language comparison.
    """
    pred_index1, _ = predict_breed(description1)
    breed_name1 = breed_mapping.get(pred_index1 + 1, "Necunoscut")
    
    pred_index2, _ = predict_breed(description2)
    breed_name2 = breed_mapping.get(pred_index2 + 1, "Necunoscut")
    
    # Retrieve some snippet from our dictionary
    desc1 = breed_descriptions.get(pred_index1 + 1, "")
    desc2 = breed_descriptions.get(pred_index2 + 1, "")
    
    # Construct a simple comparison text in Romanian
    comparison_text = f"""
Comparând cele două descrieri, prima pisică pare să fie un {breed_name1}, iar a doua un {breed_name2}.

Descriere {breed_name1}:
{desc1}

Descriere {breed_name2}:
{desc2}

Diferențele principale pot varia în funcție de aspect (blană, mărime), temperament sau nivel de energie.
"""
    return comparison_text.strip()

##########################################
# 7. MAIN
##########################################
def main():
    print("Alegeți o opțiune:")
    print("1) Generați o descriere (în limbaj natural) pentru o singură pisică.")
    print("2) Generați o comparație (în limbaj natural) între două pisici.")
    choice = input("Introdu cifra 1 sau 2: ").strip()
    
    if choice == "1":
        user_description = input("Descrieți pisica aici: ")
        result = generate_single_breed_description(user_description)
        print("\n------\n")
        print(result)
        print("\n------\n")
    elif choice == "2":
        print("Descrieți prima pisică:")
        desc1 = input(">>> ")
        print("Descrieți a doua pisică:")
        desc2 = input(">>> ")
        comparison = generate_comparison(desc1, desc2)
        print("\n------\n")
        print(comparison)
        print("\n------\n")
    else:
        print("Opțiune invalidă. Vă rugăm să rulați din nou scriptul.")

if __name__ == "__main__":
    main()
