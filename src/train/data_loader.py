import os
import gdown
import pandas as pd

# Google Drive file IDs
FILE_IDS = {
    "train": "1RFmS4oMk8eUUQ0_7rOenoxmNBuDAud1g",
    "test": "15VOfPBKZ9N7W2YCCPUx5t1eIUxuUq8so"
}

SAVE_DIR = "data/raw"

def download_from_google_drive(file_id, output_path):
    """Downloads a file from Google Drive given its file ID."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

def load_data():
    """Downloads train and test data if not present in the data/raw folder."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for dataset, file_id in FILE_IDS.items():
        file_path = os.path.join(SAVE_DIR, f"{dataset}.csv")
        if not os.path.exists(file_path):
            print(f"Downloading {dataset}.csv...")
            download_from_google_drive(file_id, file_path)
        else:
            print(f"{dataset}.csv already exists. Skipping download.")
    
    # Load Data to verify
    train_df = pd.read_csv(os.path.join(SAVE_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(SAVE_DIR, "test.csv"))
    
    print("Train and Test Data Loaded Successfully!")
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data()