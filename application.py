from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd
from logging import FileHandler,WARNING


application = Flask(__name__,template_folder='templates')
app=application
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

scaler=pickle.load(open('/config/workspace/Notebook/StandardScaler.pkl', 'rb'))
model = pickle.load(open('/config/workspace/Notebook/ModelForPrediction.pkl', 'rb'))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    output=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            output = output +'Diabetic'
        else:
            output =output + 'Non-Diabetic'
            
        return render_template('single_prediction.html',result=output)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")