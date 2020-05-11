import sys

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])


from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import re

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer , TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import pickle


def load_data(database_filepath):


    '''
    INPUT
    database_filepath - a String
    
    the function loads the data frome the database using the path 'database_filepath'
    the split the data to a data set and labels
    
    OUTPUT
    x - DataFrame 
    Y - DataFrame
    category_names  - classes names
    '''
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql("SELECT * FROM Messages", engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    return X, Y , category_names 


"""____________________________________________________________________________"""


def tokenize(text):
    '''
    INPUT
    text - a String
    
    the function will make the texte to a lower case 
    then it will delete all the special caracters 
    after that it will split it to a list of words 
    finally it will lemetize the list of words 
    
    OUTPUT
    clean_tokens - list of tokenized word 
    '''

    text = text.lower()
    text = re.sub(r"[^a-zA-z0-9]"," ",text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    
    for w in words:
        clean_tok = lemmatizer.lemmatize(w , pos='v').strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

"""___________________________________________________________________________"""

def build_model():

    '''
    INPUT
    none
    
    the function will build the model then return it
    
    OUTPUT
    model - the model
    '''

    pipeline = Pipeline([
    ('vect',TfidfVectorizer(tokenizer=tokenize)),
    ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
    'vect__smooth_idf': [True,False],
    }

    cv = GridSearchCV(pipeline,param_grid=parameters,n_jobs=-1)

"""_______________________________________________________________________"""

def evaluate_model(model, X_test, Y_test, category_names):


    '''
    INPUT
    model - the modelto evoluate
    X_test - the data to test the model
    Y_test - the classes of the data of the test
    category_names - the classes names 

    the function will compare the resuls of the model to 
    the classes of the data of test
    the prints the information and scores.
    
    OUTPUT
    none
    '''

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])

"""________________________________________________________________________"""

def save_model(model, model_filepath):

    '''
    INPUT
    model - the model to save 
    model_filepath '- the file to save the model in 

    this function will save the model in a file located in 
    model_filepath 
    
    OUTPUT
    none
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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