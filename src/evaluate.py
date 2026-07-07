from sklearn.metrics import classification_report

def evaluate(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)             #what the model predicts
    
    report = classification_report(y_test, predictions) #detailed report of precision, recall, f1 and accuracy    
    print(f"Model: {model_name}")
    print(f"{report}\n\n")
    return predictions