import pickle
from flask import Flask, render_template, request
import numpy as np


# create an flask app
app = Flask(__name__)

# Loading the model and scaler
random = pickle.load(open('car_price.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


# decorator-1
@app.route('/')
def home():
    return render_template('home.html')


# decorator-2
@app.route('/predict', methods=["GET", "POST"])
def predict():

    inputs = {}
    for key, value in request.form.items():
        if value == '' or value == ' ':
            return render_template('home.html')
        else:
            if key in ['km_driven', 'seats', 'mileage', 'engine', 'max_power', 'num_of_year']:
                inputs[key] = float(value)
            else:
                inputs[key] = value

    # Let's sepearte the categorical and numerical value for feature scaling and one-hot encoding

    num_values = []
    cat_values = {}

    for key, value in inputs.items():
        if type(value) == str:
            cat_values[key] = value
        else:
            num_values.append(value)

    # feature scaling of num_values:
    num_values = scaler.transform([num_values])

    # One_hot Encoding
    cat_features = []

    for key, value in cat_values.items():

        # fuel
        if key == 'fuel':
            if value == 'Petrol':
                cat_features.extend([0, 0, 1])
            elif value == 'Diesel':
                cat_features.extend([1, 0, 0])
            elif value == 'CNG':
                cat_features.extend([0, 0, 0])
            elif value == 'LPG':
                cat_features.extend([0, 1, 0])

        # seller-type
        if key == 'seller_type':
            if value == 'Individual':
                cat_features.extend([1, 0])
            elif value == 'Dealer':
                cat_features.extend([0, 0])
            elif value == 'Trustmark Dealer':
                cat_features.extend([0, 1])

        # transmission
        if key == 'transmission':
            if value == 'Manual':
                cat_features.extend([1])
            else:
                cat_features.extend([0])

        # owner
        if key == 'owner':
            if value == 'First Owner':
                cat_features.extend([0, 0, 0, 0])
            elif value == 'Second Owner':
                cat_features.extend([0, 1, 0, 0])
            elif value == 'Third Owner':
                cat_features.extend([0, 0, 0, 1])
            elif value == 'Fourth & Above Owner':
                cat_features.extend([1, 0, 0, 0])
            elif value == 'Test Drive Car':
                cat_features.extend([0, 0, 1, 0])

    final_features = np.concatenate([num_values, [cat_features]], axis=1)

    result = random.predict(final_features)

    return render_template('prediction.html', result=np.round(result[0], 2))


if __name__ == '__main__':
    app.run()
