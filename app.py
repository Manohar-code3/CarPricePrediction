from flask import Flask, render_template, request, redirect, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LR.pkl', 'rb'))
car = pd.read_csv('cleandata.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/get_models/<company>', methods=['GET'])
@cross_origin()
def get_models(company):
    if company == 'Select Company':
        return jsonify({'models': ['Select Model']})
    
    # Filter models based on the selected company
    models = sorted(car[car['company'] == company]['name'].unique())
    models.insert(0, 'Select Model')
    return jsonify({'models': models})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Validate inputs
    if company == 'Select Company' or car_model == 'Select Model':
        return jsonify({'error': 'Please select a valid company and model'}), 400

    try:
        prediction = model.predict(pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
        ))
        return str(np.round(prediction[0], 2))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()