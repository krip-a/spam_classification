# Spam Classification using Machine Learning

## Overview
This project classifies emails as spam or ham using Natural Language Processing (NLP) and supervised machine learning techniques

The project compares three classification algorithms:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

It uses TF-IDF vectorization to convert email text into numerical features before classification. Linear SVM achieves the best performance with an F1-score of 0.983 and was selected as the final model.

## Dataset
Dataset used:
[Kaggle Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data)

The dataset contains labeled email messages categorized as:
- Ham (not Spam) (0)
- Spam (1)

## Setup and Installation
### 1. Clone the Repository
```bash
git clone https://github.com/krip-a/spam_classification.git
cd spam_classification
```
### 2. Set Up a Virtual Environment
``` bash 
python -m venv .venv
```
    Windows: .venv\Scripts\activate
    macOS/Linux: source .venv/bin/activate
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Dataset Placement
Create a data/ directory at the root of the project and place the downloaded spam_ham_dataset.csv inside it.

## How to Run and Demo
### 1. Train and Evaluate Models
To train the models, view comparative metrics, and save the final Linear SVM model:
```bash
python main.py
```
### 2. Run Interactive Inference (Demo)
To test the saved model interactively with your own custom email strings:
```bash
python predict_app.py
```
Example Demo output:
```text
Enter email to detect spam or ham: Congratulations! You've won a free $1000 Walmart gift card. Click here to claim now!
Prediction is: 1- SPAM

Enter email to detect spam or ham: Hey, are we still meeting at the library at 3 PM?
Prediction is: 0- HAM
```

## Pipeline
1. Data Loading (`load_data.py`): Loads the raw CSV file, removes arbitrary index columns, and handles missing data.
2. Text Processing (`preprocess.py`): Executes an 80/20 train/test split and transforms raw text into mathematical vectors via TF-IDF weights.
3. Model Selection & Benchmarking (`model_selection.py`, `train.py`): Iterates through Naive Bayes, Logistic Regression, and Linear SVM to evaluate cross-performance.
4. Evaluation & Debugging (`evaluate.py`, `error_analysis.py`): Reports classification metrics and extracts misclassified samples for model transparency.
5. Serialization (`save_model.py`): Saves the vectorizer and optimal trained model as portable `.pkl` binaries.
6. Inference App (`predict_app.py`): Loads the stored artifacts to provide zero-latency, command-line predictions on unseen text inputs.

## Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization


## Models Evaluated
|                 Model | Accuracy  | Precision| Recall   |    F1    |
|-----------------------|-----------|----------|----------|----------|
|          Naive Bayes  |0.923671   |0.995370  |0.733788  |0.844794  |
|  Logistic Regression  |0.989372   |0.976351  |0.986348  |0.981324  |
|           Linear SVM  |0.990338   |0.979661  |0.986348  |0.982993  |

## Results
Linear SVM achieved the highest F1-score among the evaluated models and was selected as the final classifier

## Future Improvements
- Deploy the spam classifier using Flask or FastAPI
- Experiment with word embeddings and transformer-based models
- Perform hyperparameter tuning to improve model performance
- Add a web interface for real-time email classification