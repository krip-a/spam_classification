import joblib
from src.predict import predict_email

def main():
    model = joblib.load("saved_models/svm_model.pkl")
    vectorizer = joblib.load("saved_models/vectorizer.pkl")

    email = input("Enter email to detect spam or ham: ")

    predict_email(
        model,
        vectorizer,
        email
    )

if __name__ == "__main__":
    main()