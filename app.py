from os import name
from flask import Flask,render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import send_file

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
from pyspark import SparkConf, SparkContext


###############################################3
from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
#*** Flask configuration
 
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('static', 'uploads')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
############################################ 

spark = SparkSession\
        .builder.master("local[*]")\
        .appName("textpreprocessing")\
        .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")



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
            user = normalise(str(user))
            user = remove_punct_url_at(str(user))
            
            
            user = remove_non_ascii(str(user))
            user = remove_slashR(str(user))
            user = remove_special_char(str(user))
            user = remove_emoji(str(user))
            user = remove_multiple_space(str(user))
            user = remove_newline(str(user))
            user = replace_apostrophes(str(user))
            user = remove_multiple_comma(str(user))
            user = remove_multiple_dot(str(user))
            
            
            
        elif request.form['submit_button'] == 'pos tag':
            user = pos_tag1(user, loaded_model)
        
        elif request.form['submit_button'] == 'P & P':
            user = normalise(str(user))
            user = remove_punct_url_at(str(user))
            
            user = remove_non_ascii(str(user))
            user = remove_slashR(str(user))
            user = remove_special_char(str(user))
            user = remove_emoji(str(user))
            user = remove_multiple_space(str(user))
            user = remove_newline(str(user))
            user = replace_apostrophes(str(user))
            user = remove_multiple_comma(str(user))
            user = remove_multiple_dot(str(user))
            
            user = pos_tag1(user, loaded_model)
        
        return render_template("demo.html",user=user)

    else:
        return render_template("demo.html")
    
@app.route('/crfmodel',  methods=("POST", "GET"))
def crfmodel():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
 
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        return render_template('crfmodelsubmit.html')
    return render_template('crfmodel.html')


###CRF MODEL TRAINING
@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)
 
    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # data = uploaded_df["tagged"]
    data = uploaded_df["tagged"].apply(convert_string2_list)

    

    cutoff = int(.80 * len(data))
    training_sentences = data[:cutoff]
    test_sentences = data[cutoff:]

    X_train, y_train = transform_to_dataset(training_sentences)
    #X_test, y_test = transform_to_dataset(test_sentences)

    CRF_model_lbfgs = sklearn_crfsuite.CRF(
        algorithm = 'lbfgs',
        max_iterations = 100,
        all_possible_transitions=True,
        c1 = 0.25,
        c2 = 0.35
    )

    CRF_model_lbfgs.fit(X_train, y_train)

    # data = data.to_frame()
    # uploaded_df_html = data.to_html()

    filename = "latestCRF.joblib"
    joblib.dump(CRF_model_lbfgs, filename)
    
    return render_template('show_csv_data.html')

@app.route('/return-files/')
def return_files_tut():
	try:
		return send_file('latestCRF.joblib', as_attachment=True)
	except Exception as e:
		return str(e)

@app.route("/melexpos",  methods=("POST", "GET"))
def melexpos():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
 
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        # Retrieving uploaded file path from session
        data_file_path = session.get('uploaded_data_file_path', None)
    
        # read csv file in python flask (reading uploaded csv file from uploaded server location)
        uploaded_df = pd.read_csv(data_file_path)

        uploaded_df.rename(columns = {'Tagged_Sentence' : 'tagged'}, inplace = True)
        rdd_process = spark.sparkContext.textFile('/static/uploads/.csv')

        # data = uploaded_df["tagged"]
        data = uploaded_df["tagged"].apply(convert_string2_list)
        return render_template('melexpos2.html')
    return render_template('melexpos2.html')

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ == "__main__":
    app.run(debug=True)