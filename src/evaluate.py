from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)             #what the model predicts
    accuracy = accuracy_score(y_test, predictions)  #compare what it actually is to preductions
    print(f"Accuracy: {accuracy:.3f}")
    return predictions