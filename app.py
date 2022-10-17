from os import name
from flask import Flask,render_template
from flask import request
from flask import redirect
from flask import url_for

import sklearn
import sklearn_crfsuite
import scipy.stats
import math, string, re
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_crfsuite import CRF
from nltk.tokenize import word_tokenize
from nltk.tag.util import untag
#from sklearn.externals import joblib
import joblib
from function import *
from preprocessing import *

app = Flask(__name__)

# TESTING
@app.route("/")
def hello_world():
    name = "Andrew"
    age = 24
    occupation = "Programmer | Data Lab Research Assistant "
    return render_template('index.html', name=name, age=age, occupation=occupation)


@app.route('/demo', methods=["GET", "POST"])
def demo():
    loaded_model = joblib.load("Andrew_CRF_model.joblib")
    if request.method == "POST":
        user = request.form["nm"]
        if request.form['submit_button'] == 'preprocess':
            # user = normalise(user)
            # user = remove_punct_url_at(user)
            # user = remove_stopword_nltk(user)
            # user = unicode_problem(user)
            # user = remove_non_ascii(user)
            user = remove_slashR(user)
            user = remove_special_char(user)
            user = remove_emoji(user)
            
        elif request.form['submit_button'] == 'pos tag':
            user = pos_tag1(user, loaded_model)
        
        return render_template("demo.html",user=user)

    else:
        return render_template("demo.html")
    

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ == "__main__":
    app.run(debug=True)