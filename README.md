# Sentiment Analysis Machine Learning Pipeline

An end-to-end Machine Learning Engineering (MLE) pipeline for sentiment analysis that classifies text reviews into positive/negative categories.
 Built with Python and Docker, featuring multiple ML models and NLP techniques.

## Project Overview

Project has two main directories in source code (src/ directory). 
There are Inference and train folders. train folder contains four scripts: data_loader.py, text_processing.py, vectorization.py and train.py. 
data_loader loads the data and saves it in data/raw directory. text_processing.py processes the text body and gets it ready for vectorization part.
vectorization.py vectorizes processes data files in different vectorization methods and saves them in data/vectorized directory.
Finally, trian.py trains different models and saves serialized models in outputs/models directory. After the training is done, run_inference.py can 
be executed from src/Inference directory, which will apply the best model to the test data and save the output in the output/predictions folder. 
If specific model is preferred regardless of performance, it can manually be chosen to be ran. Instructions are in this README file. Docker volumes are provisioned, but the way commands below execute the flow, it stops the container after it runs (for stability). However, metrics and everything will appear in the logs folder. Every folder containments that will be generated via the Docker is explained below. 

All the DS part can be reviewed in the Setniment_Classification_Notebook.ipynb, where whole thought process, EDA is described. Scripts are solely focused on MLE parts (however, they do save some figures after running)

### Key Components
1. **Data Pipeline**  
   - Automatic dataset download from Google Drive
   - Text cleaning & preprocessing (HTML removal, lemmatization, etc.)
   - Multiple vectorization strategies (TF-IDF, CountVectorizer, Word2Vec)

2. **Machine Learning Models**  
   - Logistic Regression
   - Random Forest Classifier
   - Neural Network (MLP)
   - Automatic model evaluation & metric tracking

3. **MLOps Features**  
   - Dockerized training/inference environments
   - Persistent output storage
   - Logging & metrics tracking
   - Model versioning

## Project Structure
```bash
├── data/
│ ├── raw/ # Raw datasets
│ ├── processed/ # Cleaned text data
│ └── vectorized/ # Vectorized features
├── src/
│ ├── train/ # Training components
│ │ ├── data_loader.py
│ │ ├── text_processing.py
│ │ ├── vectorization.py
│ │ ├── train.py
│ │ └── Dockerfile
│ └── inference/ # Inference components
│ ├── run_inference.py
│ └── Dockerfile
├── outputs/
│ ├── models/ # Saved models
│ ├── predictions/ # Inference results
│ └── figures/ # Evaluation visualizations
└── logs/ # Processing logs
```

## Docker Execution Guide

Commands below execute whole pipeline. However, if you want container to keep running after you run it: 
Run Containers in Detached Mode (-d) Example: (docker run -d ...)
This allows the container to run in the background, so you can check logs and interact with it later.

### 1. Build Docker Images

**Training Image** (Builds model training environment):
```bash
docker build -t sentiment_train -f src/train/Dockerfile .
```

**Inference Image** (Builds model serving environment):
```bash
docker build -t sentiment_inference -f src/Inference/Dockerfile .
```

### 2. Run Training Pipeline

Executes full ML pipeline:

* Data download & preprocessing
* Feature engineering
* Model training & evaluation
* Output generation (models/metrics)

```bash
docker run --name sentiment_train_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_train
```

### 3. Run Inference

Make predictions using trained models:

```bash
docker run --name sentiment_inference_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_inference
```
Volume Mounts Explanation

* outputs/: Persists trained models and predictions
* data/: Maintains processed datasets between runs
* logs/: Stores processing logs across executions

## Key Features

### Automated Model Selection

The inference system automatically selects the best-performing model based on validation metrics.

### Output Artifacts
Location	Contents
outputs/models/	Serialized model files (.pkl, .h5)
outputs/predictions/	CSV files with prediction results
outputs/figures/	Confusion matrices & training plots
outputs/predictions/metrics.json	Model performance comparisons

### Custom Inference Options

Run specific models by name:

Run the bash and indicate the specific model name:
docker run --name sentiment_inference_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_inference --model <MODEL_NAME>

Examples below: 

LR
```bash
docker run --rm --name sentiment_inference_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_inference python /app/inference/run_inference.py --model LogisticRegression_TF-IDF
```
```bash
docker run --rm --name sentiment_inference_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_inference python /app/inference/run_inference.py --model LogisticRegression_CountVectorizer
```

RF
```bash
docker run --rm --name sentiment_inference_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_inference python /app/inference/run_inference.py --model RandomForestClassifier_TF-IDF
```
```bash
docker run --rm --name sentiment_inference_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_inference python /app/inference/run_inference.py --model RandomForestClassifier_CountVectorizer
```
MLP model
```bash
docker run --rm --name sentiment_inference_container -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data -v ${PWD}/logs:/app/logs sentiment_inference python /app/inference/run_inference.py --model MLP_Word2Vec
```
### Monitoring & Debugging

System outputs are logged. 

### Expected Output of Models:
Minimum requirement for the project was 0.85 accuracy. 

* LogisticRegression_TF-IDF: around 0.87
* LogisticRegression_CountVectorizer: around 0.88
* RandomForestClassifier_TF-IDF: around 0.85
* RandomForestClassifier_CountVectorizer: around 0.85+
* MLP_Word2Vec: around 0.86

Logically, the Inference container will choose LogisticRegression_CountVectorizer automaticallyh. 
