import os
import pandas as pd
import pickle
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Ensure output directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

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
    
    logging.info(f"Successfully loaded {file_path}")
    return df

def plot_confusion_matrix(y_true, y_pred, model_name, vectorizer_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {model_name} ({vectorizer_type})")
    plt.savefig(f"outputs/figures/{model_name.lower()}_{vectorizer_type.lower()}_confusion_matrix.png")
    plt.close()

# Load vectorized data
logging.info("Loading vectorized data...")
X_train_tfidf = load_data("data/vectorized/train_tfidf.csv").values
X_test_tfidf = load_data("data/vectorized/test_tfidf.csv").values
X_train_count = load_data("data/vectorized/train_count.csv").values
X_test_count = load_data("data/vectorized/test_count.csv").values
X_train_word2vec = load_data("data/vectorized/train_word2vec.csv").values
X_test_word2vec = load_data("data/vectorized/test_word2vec.csv").values

# Load labels
y_train = load_data("data/processed/train_cleaned.csv")["sentiment"].replace({'negative': 0, 'positive': 1})
y_test = load_data("data/processed/test_cleaned.csv")["sentiment"].replace({'negative': 0, 'positive': 1})

# Dictionary to store models and their performances
model_performance = {}

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, vectorizer_name):
    logging.info(f"Training {model.__class__.__name__} on {vectorizer_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[f"{model.__class__.__name__}_{vectorizer_name}"] = {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"{model.__class__.__name__} Accuracy ({vectorizer_name}): {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, model.__class__.__name__, vectorizer_name)
    
    # Save trained model
    model_filename = f"outputs/models/{model.__class__.__name__.lower()}_{vectorizer_name.lower()}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Saved model: {model_filename}")

# Train and Save Logistic Regression
train_and_evaluate(log_reg, X_train_tfidf, y_train, X_test_tfidf, y_test, "TF-IDF")
train_and_evaluate(log_reg, X_train_count, y_train, X_test_count, y_test, "CountVectorizer")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
train_and_evaluate(rf_model, X_train_tfidf, y_train, X_test_tfidf, y_test, "TF-IDF")
train_and_evaluate(rf_model, X_train_count, y_train, X_test_count, y_test, "CountVectorizer")

# Neural Network (MLP) for Word2Vec
logging.info("Training Neural Network on Word2Vec...")
mlp_model = Sequential([
    Input(shape=(X_train_word2vec.shape[1],)),  
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(
    optimizer=Adam(learning_rate=0.0005),  
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = mlp_model.fit(
    X_train_word2vec, y_train,
    epochs=50, batch_size=16,
    validation_data=(X_test_word2vec, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)],
    verbose=1
)

# Evaluate MLP
mlp_eval = mlp_model.evaluate(X_test_word2vec, y_test, verbose=0)
model_performance["MLP_Word2Vec"] = {
    "accuracy": mlp_eval[1],
    "loss": mlp_eval[0]
}

# Save MLP Model
mlp_model.save("outputs/models/mlp_word2vec.h5")
logging.info("Neural Network model saved.")

# Plot Training Performance
plt.figure(figsize=(10, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training vs Validation Loss")

# Save the plot
plt.savefig("outputs/figures/mlp_training_plot.png")
plt.close()

# Save Performance Summary
with open("outputs/predictions/metrics.json", "w") as f:
    json.dump(model_performance, f, indent=4)

logging.info("Model training and evaluation completed.")
