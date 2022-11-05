from flask import render_template, request, jsonify,Flask
import flask
import numpy as np
import traceback #allows you to send error to user
import pickle
import pandas as pd
import json


# App definition
app = Flask(__name__)

# importing models
with open('model.p', 'rb') as f:
   regressor = pickle.load (f)

with open('model.p', 'rb') as f:
   model_columns = pickle.load (f)

#webpage

@app.route('/')
def welcome():
   return "Welcome! Use this Flask App for Loan Medel"


@app.route('/predict', methods=['POST','GET'])
def predict():

   if flask.request.method == 'GET':
       return "Prediction page. Try using post with params to get specific prediction."

   if flask.request.method == 'POST':
       try:
           #json_ = request.get_json() # '_' since 'json' is a special word
           #print(json_)
           #json_ = json.loads(json_)
           #query_ = pd.read_json(json_)
           json_data = request.get_json() 
           query = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
           #query = query_.reindex(columns = model_columns, fill_value= 0)
           prediction = list(regressor.predict(query))

           return jsonify({
               "prediction":str(prediction)
           })

       except:
           return jsonify({
               "trace": traceback.format_exc()
               })



if __name__ == "__main__":
   app.run(debug=True, host='0.0.0.0', port=8383)

