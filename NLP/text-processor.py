import argparse
import random
import re
from collections import Counter
from langdetect import detect
from nltk import download, sent_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
download("wordnet")
download("punkt")
download("punkt_tab")

# Function to read text from file or command line
def read_text():
    parser = argparse.ArgumentParser(description="Process and analyze text.")
    parser.add_argument("--file", type=str, help="Path to the file containing the text.")
    parser.add_argument("--text", type=str, help="Text to process.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            return f.read()
    elif args.text:
        return args.text
    else:
        raise ValueError("Provide either --file or --text argument.")

# Function to identify the language of the text using langdetect
def identify_language(text):
    try:
        language = detect(text)
        return language
    except Exception as e:
        return f"Error detecting language: {e}"

# Function to perform stylometric analysis
def stylometric_analysis(text):
    words = re.findall(r'\b\w+\b', text)
    char_count = len(text)
    word_count = len(words)
    word_frequencies = Counter(words)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "word_frequencies": word_frequencies
    }

# Function to generate alternative versions of the text
def generate_alternative_versions(text):
    words = text.split()
    new_text = []
    for word in words:
        if random.random() < 0.2:  # Replace 20% of the words
            synsets = wordnet.synsets(word)
            if synsets:
                synonyms = [syn.lemmas()[0].name() for syn in synsets if syn.lemmas()]
                if synonyms:
                    new_text.append(random.choice(synonyms))
                    continue
        new_text.append(word)
    return " ".join(new_text)

# Function to extract keywords and generate sentences
def extract_keywords_and_generate_sentences(text, num_keywords=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    scores = X.toarray().sum(axis=0)
    keywords = sorted(zip(scores, feature_names), reverse=True)[:num_keywords]

    generated_sentences = {kw[1]: f"This text is about {kw[1]}." for kw in keywords}
    return generated_sentences

# Main script
if __name__ == "__main__":
    text = read_text()

    # Identify language
    lang = identify_language(text)
    print(f"Language detected: {lang}")

    # Stylometric analysis
    analysis = stylometric_analysis(text)
    print("\nStylometric Analysis:")
    print(f"Character count: {analysis['char_count']}")
    print(f"Word count: {analysis['word_count']}")
    print("Word frequencies:")
    for word, freq in analysis["word_frequencies"].items():
        print(f"{word}: {freq}")

    # Generate alternative versions
    alternative_text = generate_alternative_versions(text)
    print("\nAlternative Text:")
    print(alternative_text)

    # Extract keywords and generate sentences
    keywords_sentences = extract_keywords_and_generate_sentences(text)
    print("\nGenerated Sentences for Keywords:")
    for keyword, sentence in keywords_sentences.items():
        print(f"{keyword}: {sentence}")


# C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe ./NLP/text-processor.py --text "This is a sample text for processing."