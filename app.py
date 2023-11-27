
import pandas as pd
import numpy as np
from flask import Flask,jsonify,request,render_template
import joblib

app = Flask(__name__)

reg = joblib.load('reg_model.joblib')

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
    

@app.route('/predict',methods=['POST'])
def predict():
    if request.form:
        features = [values for values in request.form.values()]
        features = [np.array(features)]

    else:
        features = request.json
        
    query_df = pd.DataFrame(features,columns = ['area', 'bedrooms', 'bathrooms',
                                             'stories', 'mainroad', 'guestroom',
                                             'basement', 'hotwaterheating',
                                             'airconditioning', 'parking',
                                             'prefarea','furnisihing_status'])

    if query_df.loc[0,'furnisihing_status'] == 'Furnished':
        query_df['semi-furnished'] = 0
        query_df['unfurnished'] = 0

    elif query_df.loc[0,'furnisihing_status'] == 'Semi Furnished':
        query_df['semi-furnished'] = 1
        query_df['unfurnished'] = 0
    elif query_df.loc[0,'furnisihing_status'] == 'Unfurnished':
        query_df['semi-furnished'] = 0
        query_df['unfurnished'] = 1


    
    query_df[['mainroad', 'guestroom',
              'basement', 'hotwaterheating',
              'airconditioning', 'prefarea']] = query_df[['mainroad', 'guestroom',
                                                          'basement', 'hotwaterheating',
                                                          'airconditioning', 'prefarea']].replace({'Yes':1,'No':0})

    query_df.drop('furnisihing_status',axis=1,inplace=True)
    prediction = reg.predict(query_df)
    output = prediction.flatten()
    return render_template('index.html',prediction_text ="Predicted Price is : {}".format(round(output[0],2)))


if __name__=='__main__':
    app.run(debug=True)
    
