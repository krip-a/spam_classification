import joblib

def save_model(model, vectorizer):
    joblib.dump(model, "saved_models/svm_model.pkl")
    joblib.dump(vectorizer, "saved_models/vectorizer.pkl")