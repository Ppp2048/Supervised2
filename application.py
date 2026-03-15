
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

'''This initializes the Flask server.

__name__ tells Flask where the application is located.

Now app represents your web application instance. '''
application=Flask(__name__)

app=application

## Route for a home page
# Defines the homepage URL. Shows the landing page of your ML app
@app.route('/')
def index():
    return render_template('index.html') 

#This defines the prediction endpoint
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html') #This page contains the input form
    else: 
        #Flask receives input values from HTML form fields. for "POST" .
        #Flask reads user inputs from the HTML form
        #These values are passed into CustomData object.
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        
