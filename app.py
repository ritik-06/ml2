from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomeData,PredictPipeline

application = Flask(__name__)
app = application

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/',methods =['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomeData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df = data.get_data_as_dataFrame()
        predict_pipepline = PredictPipeline()
        results = predict_pipepline.predict(pred_df)
        return render_template('home.html',results = round(results[0],1))

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)