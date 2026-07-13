from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from src.predict import predict_email
from src.model_selection import get_models
from src.error_analysis import analyze_errors
from sklearn.svm import LinearSVC           #Linear SVM chosen based on highest f1-score
from src.save_model import save_model

import pandas as pd


def main():
    df = load_data()
    #X_test is original X
    X_train_vec, X_test_vec, X_test, y_train, y_test, vectorizer = preprocess(df)

    results = []
    models = get_models()       #selecting from 3 models

    # Training each model and evaluating prediction metrics
    for name, model in models.items():
        model = train(model, X_train_vec, y_train)

        _, accuracy, precision, recall, f1 = evaluate(
            model, 
            X_test_vec, 
            y_test
            )
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    
    #Linear SVM is the final model chosen
    final_model = LinearSVC()
    final_model = train(
        final_model, 
        X_train_vec,
        y_train
    )

    #Saving trained LinearSVM and vectorizer
    save_model(final_model, vectorizer)

    #Error Analysis
    predictions = final_model.predict(X_test_vec)
    analyze_errors(
        X_test,
        y_test,
        predictions
    )

    #Predicting Spam or Ham for new entered email
    new_email = input("\n\nEnter email to detect spam or ham: ")
    predict_email(
        final_model, 
        vectorizer,
        new_email
    )

if __name__ == "__main__":      #"if this file is run directly, execute main() function"
    main()