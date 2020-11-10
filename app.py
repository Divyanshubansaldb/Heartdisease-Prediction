# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart_disease_prediction_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain_type = int(request.form['chest_pain_type'])
        resting_blood_pressure = int(request.form['resting_blood_pressure'])
        cholesterol = int(request.form['cholesterol'])
        fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
        rest_ecg = int(request.form['rest_ecg'])
        max_heart_rate_achieved = int(request.form['max_heart_rate_achieved'])
        exercise_induced_angina = int(request.form['exercise_induced_angina'])
        st_depression = float(request.form['st_depression'])
        st_slope = int(request.form['st_slope'])
        num_major_vessels = int(request.form['num_major_vessels'])
        thalassemia = int(request.form['thalassemia'])
        
        data = np.array([[age,sex,chest_pain_type,resting_blood_pressure,cholesterol,fasting_blood_sugar,rest_ecg,max_heart_rate_achieved,exercise_induced_angina,st_depression,st_slope,num_major_vessels,thalassemia]])

        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)