import json
import plotly
import pandas as pd

import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine 

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

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

# load data from database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
#load model
model = joblib.load("../randomforest.pkl")

#sepecify the web adress :master.html (1st web page)
@app.route('/')
@app.route('/master')

def master():
    # distribution of genres of messages
    genre_counts = df.groupby('genre').count()['id']
    genre_names = list(genre_counts.index)
    print('genre_names:',genre_names)

    # distritbution of category of disastor
    category_counts = df[df.columns[4:]].sum()
    category_names = category_counts.index.tolist()

    figures=[
        { 'data' : [
                    Bar(x=genre_names,y=genre_counts)
                    ],

         'layout' : {
                    'title' : 'Distribution of Genres of Messages',
                    'yaxis' : {'title' : 'Count'},
                    'xaxis' : {'title' : 'Genre'}
                    }
        },

        { 'data' : [
                    Bar(x=category_names,y=category_counts)
                    ],

         'layout' : {
                    'title' : 'Distribution of Category of Disastor',
                    'yaxis' : {'title' : 'Count'},
                    'xaxis' : {'title' : 'Category'}
                    }
        }

          ]

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


#sepecify the web adress: go.html (second web page)
@app.route('/')
@app.route('/go')


# classification result page of web app
def go():
    # web page that handles user query and displays model results
    query = request.args.get('query') 

    classification_label = model.predict([query])[0]
    classification_result = dict(zip(df.columns[4:], classification_label))

    return render_template('go.html', query = query, classification_result = classification_result)
    
    
def main():
    app.run()

if __name__ == '__main__':
    main()
