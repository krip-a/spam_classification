# Spam Classification using Machine Learning

## Overview
This project classifies emails as spam or ham using natural language processing and supervised machine learning techniques

The project compares three machine learning models:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

It uses TF-IDF vectorization to convert email text into numerical features before classification

# Dataset

Dataset used:
https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data

The dataset contains leabeled email messages categorized as:
- Ham (not Spam) (0)
- Spam (1)

## Pipeline
Load Dataset ->
Train/Test split ->
TF-IDF Vectorization ->
Train models ->
Evaluate using accuracy, Precision, Recall, F1-score ->
Select best performing model ->
Predict New Emails

# Technologies
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