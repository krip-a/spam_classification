def analyze_errors(X_test, y_test, predictions):

    wrong_predictions = predictions != y_test

    for email, actual, predicted in zip(
        X_test[wrong_predictions],
        y_test[wrong_predictions],
        predictions[wrong_predictions]
    ):
        print("-" * 80)
        print("Actual: ", actual)
        print("Predicted: ", predicted)
        print("Email:")
        print(email)
