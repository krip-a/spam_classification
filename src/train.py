from models.naive_bayes import create_model

def train(X_train, y_train):
    model = create_model()
    model.fit(X_train, y_train)
    return model