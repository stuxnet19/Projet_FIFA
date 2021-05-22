# web app packages
import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
import json
# for data loading and transformation
import numpy as np 
import pandas as pd
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
# for saving/loading the ML model
import pickle


db_filename="database.db"
model_filename="models/model.pkl"
players_api = pd.read_csv('data/players_finessed.csv').set_index('Name')
teams_api = pd.read_csv('data/Clubs_finessed.csv').set_index('Club')
app=Flask(__name__)

@app.route("/")
def index():
   	return "hello world"
	#return render_template("index.html")

@app.route('/api/get_similar_players',methods=['GET'])
def get_similar_players():
	msg_data = {}
	for k in request.args.keys():
		val = request.args.get(k)
		msg_data[k] = val
		# récupérer le modéle
	
	knn = pickle.load(open('models/similar_players_knn.pickle','rb'))
	x = players_api.drop(columns=['Value']) 
	similar_players = knn.kneighbors(X=x[x.index.str.contains(str(msg_data['player']))],n_neighbors=7)
	similar_players_df = players_api.iloc[list(similar_players[1][0]),:]
	var = similar_players_df.T.to_dict()
	return jsonify(var)


@app.route('/api/get_similar_teams',methods=['GET'])
def get_similar_teams():
	msg_data = {}
	for k in request.args.keys():
		val = request.args.get(k)
		msg_data[k] = val
		# récupérer le modéle
	knn = pickle.load(open('models/similar_teams_knn.pickle','rb'))
	similar_teams = knn.kneighbors(X=teams_api[teams_api.index.str.contains(str(msg_data['team']))],n_neighbors=10)
	similar_teams_df = teams_api.iloc[list(similar_teams[1][0]),:]
	var = similar_teams_df.T.to_dict()
	return jsonify(var)



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
	app.run()
	'''
	app.debug = False
	from werkzeug.serving import run_simple
	run_simple("localhost", 5000, app)
	'''