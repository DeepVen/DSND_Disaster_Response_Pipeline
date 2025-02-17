import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')
    
# initialise Flask app
app = Flask(__name__)

def tokenize(text):
    '''
    Tokenize text data for further processing
    Args: 
    text: Messages 
    Returns:
    clean_tokens: Tokenizing and lemmatizing of text and stop words removed.
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_token = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_token)
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_source', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.drop(['id','message','original','genre'], axis=1).sum()
    category_names = category_counts.index.tolist()

    word_series = pd.Series(' '.join(df['message']).lower().split())
    stop_words = word_series.isin(stopwords.words("english"))
    top_words = word_series[~stop_words].value_counts()[:5]
    top_words_names = list(top_words.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message',
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
                    x=top_words_names,
                    y=top_words
                )
            ],

            'layout': {
                'title': 'Top 5 frequently used words',
                'yaxis': {
                    'title': "Total Count",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
         {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Category Distribution',
                'yaxis': {
                    'title': "Total Count",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    
                }
            }
        },
        
    
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