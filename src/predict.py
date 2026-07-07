def predict_email(model, vectorizer, email):
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)
    if prediction[0] == 1:
        print("Spam")
    else:
        print("Ham")