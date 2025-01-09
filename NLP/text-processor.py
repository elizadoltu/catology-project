import argparse
import random
import re
from collections import Counter
from langdetect import detect
from nltk import download, sent_tokenize, word_tokenize
from nltk.corpus import wordnet, stopwords
from rake_nltk import Rake
from transformers import pipeline # Hugging Face Transformers library pentru a genera propozitii folosindu ne de GPT-2

# Download necessary NLTK data
download("wordnet")
download("punkt")
download("stopwords")


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


def extract_keywords_and_generate_sentences(text, num_keywords=5):
    # Detect language
    detected_language = identify_language(text)

    # Get stopwords for the detected language
    try:
        if detected_language in stopwords.fileids():
            language_stopwords = stopwords.words(detected_language)
        else:
            language_stopwords = stopwords.words("english")  # Default to English if the language is unsupported
    except Exception as e:
        print(f"Error loading stopwords for {detected_language}: {e}")
        language_stopwords = stopwords.words("english")  # Fallback to English

    # Use RAKE to extract keywords
    rake = Rake(language_stopwords)
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:num_keywords]

    # Load a text generation pipeline
    text_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

    # Generate new sentences for each keyword
    generated_sentences = {}
    for phrase in ranked_phrases:
        # Generate a sentence using the phrase as a prompt
        try:
            generated = text_generator(f"{phrase}", max_length=30, num_return_sequences=1)
            new_sentence = generated[0]["generated_text"]
            generated_sentences[phrase] = new_sentence
        except Exception as e:
            print(f"Error generating sentence for {phrase}: {e}")
            generated_sentences[phrase] = f"Could not generate a sentence for {phrase}."

    return generated_sentences


# Main script
if __name__ == "__main__":
    text = read_text()

    # Open output file
    with open("output.txt", "w", encoding="utf-8") as output_file:
        # Identify language
        lang = identify_language(text)
        print(f"Language detected: {lang}")
        output_file.write(f"Language detected: {lang}\n\n")

        # Stylometric analysis
        analysis = stylometric_analysis(text)
        print("\nStylometric Analysis:")
        output_file.write("Stylometric Analysis:\n")
        output_file.write(f"Character count: {analysis['char_count']}\n")
        output_file.write(f"Word count: {analysis['word_count']}\n")
        output_file.write("Word frequencies:\n")
        for word, freq in analysis["word_frequencies"].items():
            output_file.write(f"{word}: {freq}\n")
        output_file.write("\n")

        # Generate alternative versions
        alternative_text = generate_alternative_versions(text)
        print("\nAlternative Text:")
        print(alternative_text)
        output_file.write("Alternative Text:\n")
        output_file.write(alternative_text + "\n\n")

        # Extract keywords and generate sentences
        keywords_sentences = extract_keywords_and_generate_sentences(text)
        print("\nGenerated Sentences for Keywords:")
        output_file.write("Generated Sentences for Keywords:\n")
        for keyword, sentence in keywords_sentences.items():
            print(f"{keyword}: {sentence}")
            output_file.write(f"{keyword}: {sentence}\n")
