import json
import plotly
import pandas as pd

import re
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):

    '''
    INPUT
    text - a String
    
    the function will make the texte to a loxer case 
    then it will delete all the special caracters 
    after that it will split it to a list of words 
    finally it will lemetize the list of words 
    
    OUTPUT
    a list of tokenized word 
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

# load data
engine = create_engine('sqlite:///../data/data.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/model1.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    

    top_category_count = df.iloc[:,8:].sum().sort_values(ascending=False)[1:18]
    top_category_names = list(top_category_count.index)
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_count
                )
            ],

            'layout': {
                'title': 'Top Five Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()