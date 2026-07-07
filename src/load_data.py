import pandas as pd

def load_data():
    df = pd.read_csv('data\spam_ham_dataset.csv')
    df = df.drop(columns=["Unnamed: 0"])    #dropping unnamed column
    df = df.dropna(subset = ["text"])       #dropping rows with null values in 'text' column
    return df
