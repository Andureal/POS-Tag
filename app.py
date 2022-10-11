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

app = Flask(__name__)

@app.route("/")
def hello_world():
    name = "Andrew"
    age = 24
    occupation = "Programmer | Data Lab Research Assistant "
    return render_template('index.html', name=name, age=age, occupation=occupation)


@app.route('/demo', methods=["GET", "POST"])
def demo():

    def pos_tag1(sentence, model):
        sentence = sentence_splitter(sentence)
        sentence_features = [features(sentence, index) for index in range(len(sentence))]
        return list(zip(sentence, model.predict([sentence_features])[0]))
        
    if request.method == "POST":
        user = request.form["nm"]
        user = user.lower()

        loaded_model = joblib.load("Andrew_CRF_model.joblib")
        user = pos_tag1("If u need pls rmb to request",loaded_model )
        #return redirect(url_for("demo"))
        return render_template("demo.html",user=user)

    else:
        return render_template("demo.html")
    

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ == "__main__":
    app.run(debug=True)