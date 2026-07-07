from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(df):
    X = df["text"]          #column "text" is X
    y = df["label_num"]     #column "label_num" is y; 0=ham, 1=spam

    X_train, X_test, y_train, y_test = train_test_split(    #split the data into subsets
        X,                                                  
        y,
        test_size = 0.2,                                    #80% for training, 20% for testing
        random_state = 42
        )                                  #esures data splits in the same way every time

    #TF-IDF gives higher weight to words that are important but less common across al documents
    
    #to vectorize high frequency words to detect spam
    vectorizer =  TfidfVectorizer(
        stop_words = 'english',   #stop_words = in, the, an, etc.; less importance
        lowercase = True
        )

    #learn the high frequency words from this subset and transform into vector
    X_train_vec = vectorizer.fit_transform(X_train)

    #using the recently learnt words, trainsfrom the test set
    X_test_vec = vectorizer.transform(X_test)

    return (
        X_train_vec, 
        X_test_vec, 
        y_train, 
        y_test, 
        vectorizer
    )