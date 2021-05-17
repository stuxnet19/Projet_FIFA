# web app packages
import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response

import json

# for data loading and transformation
import numpy as np 
import pandas as pd

# for statistics output
from scipy import stats
from scipy.stats import randint

# projetFifa


from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE


# for db connection
import sqlite3
db_filename="database.db"

# for saving/loading the ML model
import pickle
model_filename="models/model.pkl"

# to bypass warnings in the jupyter notebook
import warnings
from pandas.core.common import SettingWithCopyWarning


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=PendingDeprecationWarning)

app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.route("/")
def index():
   	return render_template("index.html")

@app.route('/api/get_similar_players',methods=['GET'])
def get_similar_players():
	msg_data = {}
	for k in request.args.keys():
		val = request.args.get(k)
		msg_data[k] = val
	
	return msg_data


'''

# instantiate index page



# return model predictions
@app.route("/api/predict", methods=["GET"])
def predict():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	f = open("models/X_test.json","r")
	X_test = json.load(f)
	f.close()
	all_cols=X_test
	input_df=pd.DataFrame(msg_data,columns=all_cols,index=[0])
	model = pickle.load(open(model_filename, "rb"))
	arr_results = model.predict(input_df)
	treatment_likelihood=""
	if arr_results[0]==0:
		treatment_likelihood="No"
	elif arr_results[0]==1:
		treatment_likelihood="Yes"
	return treatment_likelihood
'''
if __name__ == "__main_":
	app.debug = False
	from werkzeug.serving import run_simple
	run_simple("localhost", 5000, app)