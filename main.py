from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from src.predict import predict_email
from src.model_selection import get_models
from models.svm import create_model             #chose svm (highest f1)

import pandas as pd


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = preprocess(df)

    results = []
    models = get_models()

    for name, model in models.items():
        model = train(model, X_train, y_train)

        predictions, accuracy, precision, recall, f1 = evaluate(
            model, 
            X_test, 
            y_test, 
            name)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    final_model = create_model()
    final_model = train(
        final_model, 
        X_train,
        y_train
    )
    #new_email = input("Enter email to detect spam or ham: ")

    predict_email(
        model, 
        vectorizer,
        "Congratulations! You have won a free iPhone! click here."
    )

if __name__ == "__main__":      #"if this file is run directly, execute main() function"
    main()