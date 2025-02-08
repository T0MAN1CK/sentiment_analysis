import os
import pandas as pd
import numpy as np
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec

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

def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found!")
        raise FileNotFoundError(f"Error: File {file_path} not found!")
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        logging.error(f"{file_path} is empty!")
        raise ValueError(f"Error: {file_path} is empty!")
    
    expected_columns = {'cleaned_review', 'word2vec_review'}
    if not expected_columns.issubset(set(df.columns)):
        missing_columns = expected_columns - set(df.columns)
        logging.error(f"Missing required columns {missing_columns} in {file_path}")
        raise ValueError(f"Error: Missing required columns {missing_columns} in {file_path}")
    
    logging.info(f"Successfully loaded {file_path}")
    return df

# TF-IDF Vectorization
def apply_tfidf_vectorization(texts, max_features=5000):
    logging.info("Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(ngram_range=(1,4), max_features=max_features) 
    tfidf_matrix = vectorizer.fit_transform(texts)
    logging.info("TF-IDF Vectorization completed.")
    return tfidf_matrix, vectorizer

# CountVectorizer
def apply_count_vectorization(texts, max_features=5000):
    logging.info("Applying CountVectorizer...")
    vectorizer = CountVectorizer(ngram_range=(1,4), max_features=max_features)
    count_matrix = vectorizer.fit_transform(texts)
    logging.info("CountVectorizer completed.")
    return count_matrix, vectorizer

# Word2Vec Training
def train_word2vec(sentences, vector_size=100, window=5, min_count=2, workers=4):
    logging.info("Training Word2Vec model...")
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    logging.info("Word2Vec model training completed.")
    return model

# Convert text to Word2Vec vectors
def get_sentence_embedding(sentence, model):
    word_vectors = [model.wv[word] for word in sentence if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)  # Mean Pooling
    else:
        return np.zeros(model.vector_size)  # Zero vector if no words are found

# Main Execution
if __name__ == "__main__":
    try:
        os.makedirs("data/vectorized", exist_ok=True)
        
        logging.info("Loading train data...")
        train_df = load_data("data/processed/train_cleaned.csv")
        logging.info("Loading test data...")
        test_df = load_data("data/processed/test_cleaned.csv")
        
        # TF-IDF Vectorization
        tfidf_train, tfidf_vectorizer = apply_tfidf_vectorization(train_df['cleaned_review'])
        tfidf_test = tfidf_vectorizer.transform(test_df['cleaned_review'])
        pd.DataFrame(tfidf_train.toarray()).to_csv("data/vectorized/train_tfidf.csv", index=False)
        pd.DataFrame(tfidf_test.toarray()).to_csv("data/vectorized/test_tfidf.csv", index=False)
        
        # CountVectorizer
        count_train, count_vectorizer = apply_count_vectorization(train_df['cleaned_review'])
        count_test = count_vectorizer.transform(test_df['cleaned_review'])
        pd.DataFrame(count_train.toarray()).to_csv("data/vectorized/train_count.csv", index=False)
        pd.DataFrame(count_test.toarray()).to_csv("data/vectorized/test_count.csv", index=False)
        
        # Word2Vec Training
        train_sentences = train_df['word2vec_review'].apply(str.split)
        test_sentences = test_df['word2vec_review'].apply(str.split)
        
        word2vec_model = train_word2vec(train_sentences)
        
        # Convert entire dataset into vectors
        train_vectors = np.array([get_sentence_embedding(sentence, word2vec_model) for sentence in train_sentences])
        test_vectors = np.array([get_sentence_embedding(sentence, word2vec_model) for sentence in test_sentences])
        
        # Save Word2Vec vectors
        pd.DataFrame(train_vectors).to_csv("data/vectorized/train_word2vec.csv", index=False)
        pd.DataFrame(test_vectors).to_csv("data/vectorized/test_word2vec.csv", index=False)
        
        logging.info("Vectorization completed! Vectorized data saved.")
    except Exception as e:
        logging.error(f"Vectorization failed: {e}")
        print(f"Error: {e}")
