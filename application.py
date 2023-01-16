from flask import Flask, request, render_template
import joblib
import keras
import numpy

application = Flask(__name__)

XGBmodel = joblib.load('XGBmodel.pkl')
XGBscaler = joblib.load('XGBscaler.pkl')

'''
Neuralmodel = keras.models.load_model('C:\\Users\\bence\\Documents\\python programs\\cancer-predicting-models')
Neuralscaler = joblib.load('Neuralscaler.pkl')
'''

@application.route("/")
def home():
    return render_template('index.html')

@application.route("/index.html")
def index():
    return render_template('index.html')

@application.route("/pages/about.html")
def about():
    return render_template('/pages/about.html')

@application.route('/predict', methods=['POST'])
def predict():

    features = request.form.to_dict()

    if (len(features) == 32):
        features = list(features.values())
        features = list(map(int, features))

        # bmi calculation
        feet = features[1]
        inches = features[2]
        converted_inches = inches / 12
        feet += converted_inches
        meters = feet / 3.281
        pounds = features[3]
        kg = pounds / 2.205

        bmi = 27.282130617023647
        if (meters != 0):
            bmi = kg / (pow(meters,2))
            
        del features[1:4]
        features.insert(14, bmi)
        print(features)

        features = XGBscaler.transform([features])

        prediction_result = XGBmodel.predict(features)
        prediction_probability = XGBmodel.predict_proba(features)

        if (prediction_result == 0):
            return render_template('index.html', prediction = 'The model is '+str((prediction_probability[0][0])*100)+'% confident that you DO NOT have colon cancer. Please remember that this is not a diagnosis.')
            
        if (prediction_result == 1):
            return render_template('index.html', prediction = 'The model is '+str((prediction_probability[0][1])*100)+'% confident that you should get a screening for colon cancer. Please remember that this is not a diagnosis.')
        
    else:
        return render_template('index.html', prediction = 'Failed')
    
if __name__ == '__main__':
    application.run(debug=True)