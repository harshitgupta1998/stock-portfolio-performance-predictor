# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 00:18:18 2020

@author: harshit
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app=Flask(__name__)
model1=pickle.load(open('model1.pkl','rb'))
model2=pickle.load(open('model2.pkl','rb'))
model3=pickle.load(open('model3.pkl','rb'))
model4=pickle.load(open('model4.pkl','rb'))
model5=pickle.load(open('model5.pkl','rb'))
model6=pickle.load(open('model6.pkl','rb'))
@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction1 = model1.predict(final_features)
        prediction2 = model2.predict(final_features)
        prediction3 = model3.predict(final_features)
        prediction4 = model4.predict(final_features)
        prediction5 = model5.predict(final_features)
        prediction6 = model6.predict(final_features)

        output1 = round(prediction1[0], 2)
        output2 = round(prediction2[0], 2)
        output3 = round(prediction3[0], 2)
        output4 = round(prediction4[0], 2)
        output5 = round(prediction5[0], 2)
        output6 = round(prediction6[0], 2)
        
        return render_template('index.html', prediction_text='Your Portfolio is attributed to {}% annual return and Excess return of {}%  with Systematic Risk {} and Total Risk of {}% having absolute Win Rate {}% with probable Relative Win Rate {}% '.format(output1,output2,output3,output4,output5,output6 ))
            
if __name__ == "__main__":
    app.run(debug=True)