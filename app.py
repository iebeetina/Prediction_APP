from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_adoption', methods=['POST', 'GET'])
def predict_adoption():
    # get the parameters
    
    persons_age = int(request.form['age'])
    persons_it_experience = int(request.form['it_experience'])
    persons_country = int(request.form['country'])
    persons_primary_language = str(request.form['primary_language'])



    # load the X_columns file
    X_columns = joblib.load('model/X_columns.joblib')
    print(X_columns)

    # generate a dataframe with zeros
    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    print(df_prediction)

    # change the dataframe according to the inputs
    
    df_prediction.at[0, 'age'] = persons_age
    df_prediction.at[0, 'country'] = persons_country
    df_prediction.at[0, 'it.experience'] = persons_it_experience
    df_prediction.at[0, 'programming.language.primary.'+str(persons_primary_language)] = 1.0
    print(df_prediction)

    # load the model and predict
    model = joblib.load('model/model.joblib')
    prediction = model.predict(df_prediction.head(1).values)
    predicted_job = prediction

    if persons_age == 0:
      persons_age = 'between 21 and 29'
    elif persons_age == 1:
      persons_age = 'between 40 and 49'
    elif persons_age == 2:
      persons_age = 'between 30 and 39'
    elif persons_age == 3:
      persons_age = '17 years or younger'
    elif persons_age == 4:
      persons_age = 'between 18 and 20' 
    elif persons_age == 5:
      persons_age = 'between 50 and 59'
    elif persons_age == 6:
      persons_age = '60 years or older'
            

    if persons_it_experience == 1:
      persons_it_experience = 'less than one year'
    elif persons_it_experience == 2:
      persons_it_experience = '1 to 2 years'
    elif persons_it_experience == 3:
      persons_it_experience = '2 to 4 years'
    elif persons_it_experience == 4:
      persons_it_experience = '3 to 5 years'
    elif persons_it_experience == 8:
      persons_it_experience = '6 to 10 years'
    elif persons_it_experience == 11:
      persons_it_experience = '11 or more years'


    if persons_primary_language == 'Java':
      persons_primary_language = 'Java'
    elif persons_it_experience == 'C':
      persons_primary_language = 'C'
    elif persons_primary_language == 'C++':
      persons_primary_language = 'C++'
    elif persons_primary_language == 'Python':
      persons_primary_language = 'Python'  
    elif persons_it_experience == 'C#':
      persons_primary_language = 'C#'
    elif persons_primary_language == 'PHP':
      persons_primary_language = 'PHP'
    elif persons_it_experience == 'JavaScript':
      persons_primary_language = 'JavaScript'
    elif persons_primary_language == 'Scala':
      persons_primary_language = 'Scala'
    elif persons_primary_language == 'R':
      persons_primary_language = 'R'  
    
     
   






    
    
    

    if predicted_job == 'job.app.type.Web Back-end':
      predicted_job = 'as Web Back-end Developer'
    elif predicted_job == 'job.app.type.Web Front-end':
      predicted_job = 'as Web Front-end Developer'
    elif predicted_job == 'job.app.type.Mobile applications':
      predicted_job = 'as Mobile application Developer'
    elif predicted_job == 'job.app.type.Desktop':
      predicted_job = 'as Desktop Applications Developer'
    elif predicted_job == 'job.app.type.Other Back-end':
      predicted_job = ' as Back-end developer other than Web'
    elif predicted_job == 'job.app.type.Data analysis':
      predicted_job = 'in Data analysis'
    elif predicted_job == 'job.app.type.BI':
      predicted_job = 'in Business Intellegance'
    elif predicted_job == 'job.app.type.Machine learning':
      predicted_job = ' in Machine learning'
    elif predicted_job == 'job.app.type.Libraries / Frameworks':
      predicted_job = 'as Libraries / Frameworks Developer'
    elif predicted_job == 'job.app.type.Embedded / IoT':
      predicted_job = 'as Embedded / IoT Developer'
    elif predicted_job == 'job.app.type.Other - Write In':
      predicted_job = 'Other'


    return render_template('results.html',
                           persons_age=str(persons_age),
                           persons_it_experience=str(persons_it_experience),
                           predicted_job=str(predicted_job),
                           persons_primary_language=str(persons_primary_language)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
