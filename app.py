# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'covid-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Breathing = int(request.form['Breathing'])
        Fever = int(request.form['Fever'])
        Dry_Cough = int(request.form['Dry_Cough'])
        Sore_throat = int(request.form['Sore_throat'])
        Running_Nose = int(request.form['Running_Nose'])
        Asthma = int(request.form['Asthma'])
        Chronic_Lung_Disease = int(request.form['Chronic_Lung_Disease'])
        Headache = int(request.form['Headache'])
        Heart_Disease = int(request.form['Heart_Disease'])
        Diabetes = int(request.form['Diabetes'])
        Hyper_Tension = int(request.form['Hyper_Tension'])
        Fatigue = int(request.form['Fatigue'])
        Gastrointestinal = int(request.form['Gastrointestinal'])
        Abroad_travel = int(request.form['Abroad_travel'])
        Contact_COVID_Patient = int(request.form['Contact_COVID_Patient'])
        Attended_Large_Gathering = int(request.form['Attended_Large_Gathering'])
        Visited_Public_Exposed_Places = int(request.form['Visited_Public_Exposed_Places'])
        Family_working_Public_Exposed_Places = int(request.form['Family_working_Public_Exposed_Places'])
        Wearing_Masks = int(request.form['Wearing_Masks'])
        Sanitization_Market = int(request.form['Sanitization_Market'])

        data = np.array([[Breathing, Fever, Dry_Cough, Sore_throat, Running_Nose, Asthma, Chronic_Lung_Disease,
                           Headache, Heart_Disease, Diabetes, Hyper_Tension, Fatigue, Gastrointestinal, Abroad_travel,
                           Contact_COVID_Patient, Attended_Large_Gathering, Visited_Public_Exposed_Places,
                           Family_working_Public_Exposed_Places, Wearing_Masks, Sanitization_Market]])

        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=True)


