import os
import re
import string
import unicodedata
import pandas as pd
import contractions
import emoji
import nltk
import wordninja
import logging
import sys
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure logs directory exists before configuring logging
os.makedirs("logs", exist_ok=True)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log"),  # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataProcessingError(Exception):
    """Custom exception for errors during text processing."""
    pass

# Load dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found!")
        raise FileNotFoundError(f"Error: File {file_path} not found!")
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        logging.error(f"{file_path} is empty!")
        raise DataProcessingError(f"Error: {file_path} is empty!")
    
    expected_columns = {'review'}
    if not expected_columns.issubset(set(df.columns)):
        missing_columns = expected_columns - set(df.columns)
        logging.error(f"Missing required columns {missing_columns} in {file_path}")
        raise DataProcessingError(f"Error: Missing required columns {missing_columns} in {file_path}")
    
    if df['review'].isnull().sum() > 0:
        logging.warning("Null values found in the 'review' column!")
    
    if not all(isinstance(x, str) for x in df['review'].dropna()):
        logging.error("Non-string values detected in 'review' column!")
        raise DataProcessingError("Error: Non-string values detected in 'review' column!")
    
    logging.info(f"Successfully loaded {file_path}")
    return df

# Remove Duplicates
def remove_duplicates(df):
    before_count = df.shape[0]
    df = df.drop_duplicates(subset=['review']).reset_index(drop=True)
    after_count = df.shape[0]
    logging.info(f"Removed {before_count - after_count} duplicate reviews.")
    return df

# Text Processing Auxiliary Functions

def to_lowercase(text):
    return text.lower()

def clean_html_urls(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)  # Remove URLs
    text = unicodedata.normalize("NFKD", text)  # Normalize unicode 
    return text

def clean_text(text):
    text = re.sub(r"(\b\w+(?:\s+\w+)*\b)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    text = re.sub(r'\b(?:[a-zA-Z]{3,}\s*){3,}\b', '', text)
    return text 

def expand_contractions(text):
    return contractions.fix(text)

def normalize_hyphens(text):
    text = text.replace("-", " ")  
    return text

def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def split_merged_words(text):
    words = text.split()
    words = [" ".join(wordninja.split(word)) if len(word) > 8 else word for word in words]  
    return " ".join(words)

def remove_punctuation_numbers(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = re.sub(r'\d+', '', text)  
    return text

def remove_repeated_phrases(text):
    """Remove sequences of repeated words and phrases in text efficiently."""
    pattern = re.compile(r"\b(\w+(?:\s+\w+){0,5})\b(?:\s+\1\b)+", re.IGNORECASE)
    return pattern.sub(r"\1", text)  

def tokenize(text):
    return word_tokenize(text)


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def remove_short_words(tokens):
    return [word for word in tokens if len(word) > 2]

def tokens_to_string(tokens):
    return " ".join(tokens) # For Word2Vec-ready text 


nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# Function to apply lemmatization
def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(df):
    logging.info("Starting text preprocessing...")
    logging.info("Removing duplicates...")
    df = remove_duplicates(df)
    logging.info("Converting to lowercase...")
    df['cleaned_review'] = df['review'].apply(to_lowercase)
    logging.info("Removing HTML and URLs...")
    df['cleaned_review'] = df['cleaned_review'].apply(clean_html_urls)
    #logging.info("Cleaning text...")
    #df['cleaned_review'] = df['cleaned_review'].apply(clean_text)
    logging.info("Expanding contractions...")
    df['cleaned_review'] = df['cleaned_review'].apply(expand_contractions)
    logging.info("Normalizing hyphens...")
    df['cleaned_review'] = df['cleaned_review'].apply(normalize_hyphens)
    logging.info("Converting emojis...")
    df['cleaned_review'] = df['cleaned_review'].apply(convert_emojis)
    logging.info("Splitting merged words...")
    df['cleaned_review'] = df['cleaned_review'].apply(split_merged_words)
    logging.info("Removing punctuation and numbers...")
    df['cleaned_review'] = df['cleaned_review'].apply(remove_punctuation_numbers)
    logging.info("Removing repeated phrases...")
    df['cleaned_review'] = df['cleaned_review'].apply(remove_repeated_phrases)
    logging.info("Tokenizing text...")
    df['tokens'] = df['cleaned_review'].apply(tokenize)
    logging.info("Removing stopwords...")
    df['tokens'] = df['tokens'].apply(remove_stopwords)
    logging.info("Removing short words...")
    df['tokens'] = df['tokens'].apply(remove_short_words)
    logging.info("Saving before lemmatization for Word2Vec Vectorization...")
    df['word2vec_review'] = df['tokens'].apply(tokens_to_string)
    logging.info("Applying lemmatization...")
    df['lemmatized_tokens'] = df['tokens'].apply(apply_lemmatization)
    logging.info("Converting lemmatized data back to string")
    df['cleaned_review'] = df['lemmatized_tokens'].apply(tokens_to_string)
    logging.info("Text preprocessing completed.")
    print(df.head())  # Check if the DataFrame is being modified correctly

    return df

# Main Execution
if __name__ == "__main__":
    try:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        logging.info("Loading train data...")
        train_df = load_data("data/raw/train.csv")
        logging.info("Loading test data...")
        test_df = load_data("data/raw/test.csv")
        
        logging.info("Preprocessing train data...")
        train_df = preprocess_text(train_df)
        if train_df.empty or test_df.empty:
            logging.error("Error: Processed DataFrame is empty after preprocessing!")
            print("Error: Processed DataFrame is empty after preprocessing!")

        logging.info("Preprocessing test data...")
        test_df = preprocess_text(test_df)
        if train_df.empty or test_df.empty:
            logging.error("Error: Processed DataFrame is empty after preprocessing!")
            print("Error: Processed DataFrame is empty after preprocessing!")

        print("Final Train DataFrame Shape:", train_df.shape)
        print("Final Test DataFrame Shape:", test_df.shape)

        logging.info("Saving cleaned train data...")
        train_df.to_csv("data/processed/train_cleaned.csv")
        logging.info("Saving cleaned test data...")
        test_df.to_csv("data/processed/test_cleaned.csv")
        
        logging.info("Preprocessing completed! Cleaned data saved.")
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        print(f"Error: {e}")  
