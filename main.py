from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from src.predict import predict_email

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = preprocess(df)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
    print()

    predict_email(
        model, 
        vectorizer,
        "Congratulations! You have won a free iPhone! click here."
    )

if __name__ == "__main__":      #"if this file is run directly, execute main() function"
    main()