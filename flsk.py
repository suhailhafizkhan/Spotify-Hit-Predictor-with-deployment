# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:54:22 2020

@author: Zoheb
"""
from wsgiref import simple_server
import flask
import numpy as np
#!pip install flask
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model_pred.pkl', 'rb'))
scalar=pickle.load(open("Standard_scaler.pkl",'rb'))
pca_model=pickle.load(open('pca_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    new_df=pd.DataFrame(final_features)
    scaled_data = scalar.transform(new_df)
    principal_data = pca_model.transform(scaled_data)
    principalsx=pd.DataFrame(principal_data,columns=["PC-1","PC-2","PC-3","PC-4","PC-5","PC-6","PC-7","PC-8","PC-9","PC-10"])
    prediction = model.predict(principalsx)
    if prediction[0] == 0:
        result = "Flop"
    else:
        result = "Hit"
    
    return result

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(result))



if __name__ == "__main__":
    app.run(debug=True)

