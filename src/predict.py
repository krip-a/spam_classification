def predict_email(model, vectorizer, email):
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)
    if prediction[0] == 1:
        print("Prediction is: 1- SPAM")
    else:
        print("Prediction is: 0- HAM")