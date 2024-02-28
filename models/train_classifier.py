import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Load data from SQLite
    INPUT:
        database_filepath: path of database
    OUTPUT:
        X: data contain message value
        Y: data use for training model
        category_names:
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X, Y, category_names 


def tokenize(text):
    """
    INPUT: 
        text: Input text to be tokenized.
    OUTPUT:
        clean_tokens: A list of cleaned tokens (words) from the input text.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline for multi-output classification using Random Forest.

    Returns:
        model: A grid search model with the best hyperparameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # set best parameter
    parameters = {
        'clf__estimator__n_estimators': [150],
        'clf__estimator__min_samples_split': [2]
    }
    model = GridSearchCV(pipeline, param_grid = parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a multi-output classification model using test data.

    INPUT:
        model: A trained multi-output classification model.
        X_test: Test features.
        Y_test: True labels for each category.
        category_names: List of category names.

    OUTPUT:
        Print Classification report for each category.

    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(Y_test.columns[i], ':')
        print(classification_report(list(Y_test.values[:,i]), list(Y_pred[:,i])))
        print('-----------------------------------')


def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file.

    INPUT:
        model: A trained model to be saved.
        model_filepath: Path to the file where the model will be saved.

    OUTPUT:
        None
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """
    Main function for training and evaluating a disaster response classifier.

    Reads data from a specified database file, splits it into training and testing sets,
    builds a machine learning model, trains the model, evaluates its performance,
    and saves the trained model to a pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
#         print('X_train:', X_train)
#         print('Y_train', Y_train)
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()