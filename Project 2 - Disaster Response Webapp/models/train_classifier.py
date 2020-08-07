import sys
# import libraries
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import pickle
#from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")



def load_data(database_filepath):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df.loc[:,'message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns

    return X,Y, category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)

    #Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return clean_tokens



def build_model():
    pipeline = Pipeline([
                        ('vec', CountVectorizer(tokenizer = tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs = 6)))
                        ])
    
    # can tuning the parameters, but take really long time
    #grid search
    #parameters = {'clf__estimator__max_features': ['sqrt',0.5],
    #          'clf__estimator__criterion': ['gini','entropy']
    #             }

    #cv = GridSearchCV(pipeline, param_grid = parameters, cv = 5,n_jobs=6)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_predict = model.predict(X_test)
    
    # iterate columns to get reports
    for i in range(len(Y_test.columns)):
        print(classification_report(Y_test.iloc[:, i].values, y_predict[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    #joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)
        
        print('Building model...')
        model = build_model()
        
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