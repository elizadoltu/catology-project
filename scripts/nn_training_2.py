import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

# NEW: Gensim is used to load pretrained word embeddings (GloVe)
import gensim.downloader as api

# NEW: NLTK imports for synonyms & text processing
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

# Make sure these NLTK downloads are done once in your environment
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1) LOADING THE GLOVE MODEL
print("Loading GloVe model. This may take a moment...")
glove_model = api.load("glove-wiki-gigaword-50")  # 50-dimension GloVe vectors
print("GloVe model loaded successfully!")

######################################
# 2. DEFINE SYNONYMS & EMBEDDING FUNCS
######################################
def expand_synonyms(word):
    """
    Return a set of synonyms (lemmas) for the given word using WordNet.
    - We do not hardcode synonyms ourselves; WordNet will do it dynamically.
    - Each 'synset' is a set of synonyms for a given sense of the word.
    - We gather synonyms to capture more variety in user text or training data.
    """
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            # lemma.name() might contain underscores, so we replace them with spaces
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def get_text_embedding(tokens):
    """
    Convert a list of tokens into a single embedding vector by averaging GloVe embeddings:
    - If a token is in GloVe's vocabulary, we fetch the 50-dim vector from 'glove_model'.
    - If no tokens are valid, we return a zero vector of length 50.
    - Embeddings allow our model to capture semantic similarities between words.
    """
    vectors = []
    for t in tokens:
        if t in glove_model:
            vectors.append(glove_model[t])
    if not vectors:
        return np.zeros(glove_model.vector_size)  # e.g. 50-dim zero vector
    # We average all token vectors to produce a single embedding for the entire text.
    return np.mean(vectors, axis=0)

def process_text(description):
    """
    1) Tokenize & remove stopwords.
    2) Expand each token with synonyms (limit to 2 synonyms to avoid huge expansions).
    3) Generate a 50-dimensional GloVe embedding by averaging token embeddings.
    
    This function returns a single 50-dim vector representing the textual description.
    """
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(str(description).lower())
    filtered_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]
    
    expanded_tokens = []
    for token in filtered_tokens:
        expanded_tokens.append(token)
        syns = expand_synonyms(token)
        # Limit expansions to 2 synonyms to avoid exploding the token list
        expanded_tokens.extend(list(syns)[:2])
    
    embedding = get_text_embedding(expanded_tokens)  # shape (50,)
    return embedding

####################################
# 3. ACTIVATION / LOSS FUNCTIONS
####################################
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cross_entropy_loss(y_true, y_pred):
    """
    Standard cross-entropy loss for multi-class classification.
    We clip y_pred to avoid log(0).
    """
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(np.sum(y_true * np.log(y_pred.T), axis=1))

def accuracy(y_true, y_pred):
    y_pred_labels = np.argmax(y_pred, axis=0)
    y_true_labels = np.argmax(y_true, axis=1)
    return np.mean(y_true_labels == y_pred_labels)

##########################################
# 4. PARAMETER INITIALIZATION
##########################################
def initialize_parameters(input_size, hidden_layer_size1, hidden_layer_size2, output_size):
    """
    We initialize the weights randomly within a small uniform range and biases as zeros.
    - input_size is now (k + 50) if we combine numeric columns + a 50-dim embedding.
    - hidden_layer_size1, hidden_layer_size2 are user-defined.
    - output_size is the number of classes after one-hot encoding the target.
    """
    np.random.seed(42)
    weights = {
        "W1": np.random.uniform(-0.05, 0.05, (hidden_layer_size1, input_size)),
        "W2": np.random.uniform(-0.05, 0.05, (hidden_layer_size2, hidden_layer_size1)),
        "W3": np.random.uniform(-0.05, 0.05, (output_size, hidden_layer_size2)),
    }
    biases = {
        "b1": np.zeros((hidden_layer_size1, 1)),
        "b2": np.zeros((hidden_layer_size2, 1)),
        "b3": np.zeros((output_size, 1)),
    }
    return weights, biases

##############################################
# 5. FORWARD PROPAGATION WITH DROPOUT
##############################################
def forward_propagation(X, weights, biases, keep_prob=0.8):
    """
    X shape: (m, input_size), where m is batch size (or dataset size in full-batch).
    - We do dropout on the first hidden layer to reduce overfitting.
    - A1, A2, A3 are the outputs (activations) of each layer.
    """
    Z1 = np.dot(weights["W1"], X.T) + biases["b1"]  # (hidden1, m)
    A1 = sigmoid(Z1)
    
    # Dropout on hidden layer 1
    dropout_mask = np.random.rand(*A1.shape) < keep_prob
    A1 *= dropout_mask
    A1 /= keep_prob
    
    Z2 = np.dot(weights["W2"], A1) + biases["b2"]  # (hidden2, m)
    A2 = sigmoid(Z2)
    
    Z3 = np.dot(weights["W3"], A2) + biases["b3"]  # (output, m)
    A3 = softmax(Z3)
    
    cache = {"Z1": Z1, "A1": A1,
             "Z2": Z2, "A2": A2,
             "Z3": Z3, "A3": A3}
    return A3, cache

##############################################
# 6. BACKWARD PROPAGATION (INC. DROPOUT)
##############################################
def backward_propagation(X, y, weights, biases, cache, learning_rate, lambda_reg=0.01):
    """
    We compute partial derivatives of the loss wrt W1, W2, W3 and b1, b2, b3.
    - L2 regularization is partially integrated by adding (lambda_reg / m)*weights["Wn"].
    - The final updated weights/biases are stored back in 'weights' and 'biases'.
    """
    m = X.shape[0]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    
    delta_k = A3 - y.T  # shape (output_size, m)
    delta_h2 = A2 * (1 - A2) * np.dot(weights["W3"].T, delta_k)
    delta_h1 = A1 * (1 - A1) * np.dot(weights["W2"].T, delta_h2)
    
    # Include L2 regularization in dW
    dW3 = np.dot(delta_k, A2.T) + (lambda_reg / m) * weights["W3"]
    dW2 = np.dot(delta_h2, A1.T) + (lambda_reg / m) * weights["W2"]
    dW1 = np.dot(delta_h1, X) + (lambda_reg / m) * weights["W1"]
    
    db3 = np.sum(delta_k, axis=1, keepdims=True)
    db2 = np.sum(delta_h2, axis=1, keepdims=True)
    db1 = np.sum(delta_h1, axis=1, keepdims=True)
    
    weights["W3"] -= (learning_rate * dW3) / m
    weights["W2"] -= (learning_rate * dW2) / m
    weights["W1"] -= (learning_rate * dW1) / m
    
    biases["b3"] -= (learning_rate * db3) / m
    biases["b2"] -= (learning_rate * db2) / m
    biases["b1"] -= (learning_rate * db1) / m
    
    return weights, biases

###########################################
# 7. TRAIN THE NETWORK (FULL-BATCH)
###########################################
def train_network(X_train, y_train, weights, biases, learning_rate, epochs):
    """
    A simple training loop:
    - forward_propagation
    - cross_entropy_loss
    - backward_propagation
    - keep track of the loss
    """
    losses = []
    for epoch in range(epochs):
        A3, cache = forward_propagation(X_train, weights, biases)
        loss = cross_entropy_loss(y_train, A3)
        losses.append(loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        weights, biases = backward_propagation(
            X_train, y_train, weights, biases, cache, learning_rate
        )
    return weights, biases, losses

#######################################
# 8. MAIN TRAINING LOGIC
#######################################
def main():
    # Path to your dataset
    data_path = "./data/Updated_Cats_database.xlsx"
    data = pd.read_excel(data_path)
    
    # Drop only truly irrelevant columns; keep 'Observations' for text
    data_cleaned = data.drop(columns=['Row.names', 'Timestamp'], errors='ignore')
    
    # Our text column is "Observations"
    text_column = 'Observations'
    
    # We treat everything except 'Breed' and 'Observations' as numeric features
    numeric_df = data_cleaned.drop(columns=['Breed', text_column], errors='ignore')
    X_numeric = numeric_df.values  # shape (m, k)
    
    # Convert each row's textual description into a 50-d embedding
    text_embeddings = []
    for desc in data_cleaned[text_column].fillna(""):
        emb = process_text(desc)  # shape (50,)
        text_embeddings.append(emb)
    text_embeddings = np.array(text_embeddings)  # shape (m, 50)
    
    # Combine numeric + embeddings => final X with shape (m, k+50)
    X = np.hstack((X_numeric, text_embeddings))
    
    # Encode target (Breed) using OneHotEncoder
    y = data_cleaned['Breed']
    one_hot_encoder = OneHotEncoder()
    y_encoded = one_hot_encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
    
    # Scale everything (k+50 features)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Over-sample minority classes with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    
    # input_size = (k + 50)
    input_size = X_train.shape[1]
    hidden_layer_size1 = 10
    hidden_layer_size2 = 10
    output_size = y_encoded.shape[1]
    
    learning_rate = 0.01
    epochs = 500
    
    # Initialize parameters
    weights, biases = initialize_parameters(
        input_size, hidden_layer_size1, hidden_layer_size2, output_size
    )
    
    # Train
    weights, biases, losses = train_network(
        X_train, y_train, weights, biases, learning_rate, epochs
    )
    
    # Plot training loss
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    # Evaluate on test
    A3_test, _ = forward_propagation(X_test, weights, biases)
    test_loss = cross_entropy_loss(y_test, A3_test)
    test_acc = accuracy(y_test, A3_test) * 100
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # SAVE MODEL TO DISK
    # Why pickle? We want to persist Python objects (weights, biases, scaler)
    # so we can load them later without retraining.
    with open("model_weights.pkl", "wb") as f:
        pickle.dump(weights, f)
    with open("model_biases.pkl", "wb") as f:
        pickle.dump(biases, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()
