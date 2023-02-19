from flask import Flask, request, render_template
import joblib
import json
# import keras
# import numpy

keys = [

    'age',
    'height1',
    'height2',
    'weight',
    'sex',
    'race7',
    'educat',
    'marital',
    'occupat',
    'cig_stat',
    'cig_years',
    'cigpd_f',
    'cigar',
    'pipe',
    'fh_cancer',
    'colo_fh',
    'colo_fh_cnt',
    'asppd',
    'ibuppd',
    'arthrit_f',
    'bronchit_f',
    'colon_comorbidity',
    'diabetes_f',
    'divertic_f',
    'emphys_f',
    'gallblad_f',
    'hearta_f',
    'hyperten_f',
    'liver_comorbidity',
    'osteopor_f',
    'polyps_f',
    'stroke_f'

]

XGBmodel = joblib.load('XGBmodel.pkl')
XGBscaler = joblib.load('XGBscaler.pkl')

'''
Neuralmodel = keras.models.load_model('C:\\Users\\bence\\Documents\\python programs\\cancer-predicting-models')
Neuralscaler = joblib.load('Neuralscaler.pkl')
'''

def calculateBMI(features):
    feet = features[1]
    inches = features[2]
    converted_inches = inches / 12
    feet += converted_inches

    meters = feet / 3.281
    if (meters == 0): return 27.282130617023647 # division by zero if unchecked

    pounds = features[3]
    kg = pounds / 2.205
    bmi = kg / (pow(meters,2)) # division

    return bmi

application = Flask(__name__)

@application.route("/")
def home():
    return render_template('index.html')

@application.route("/index.html")
def index():
    return render_template('index.html')

@application.route("/pages/about.html")
def about():
    return render_template('/pages/about.html')

@application.route("/pages/colon.html")
def colon():
    return render_template('/pages/colon.html')

@application.route("/pages/pancreatic.html")
def pancreatic():
    return render_template('/pages/pancreatic.html')

@application.route("/pages/lung.html")
def lung():
    return render_template('/pages/lung.html')

@application.route("/pages/api.html")
def api():
    return render_template('/pages/api.html')

@application.route('/api', methods=['POST'])
def predict():

    payloadType = request.headers.get('Content-Type')
    if (payloadType == 'application/json'):

        data = request.get_json() 
        userKeys = []

        for key in data: userKeys.append(key)
        
        if (userKeys == keys):

            features = []
            for value in data.values():
                try: float(value)
                except: return 'Payload values not of type int or float'
                features.append(float(value))
                
            bmi = calculateBMI(features)
            del features[1:4]
            features.insert(14, bmi)

            features = XGBscaler.transform([features])
            prediction = XGBmodel.predict(features)
            probability = XGBmodel.predict_proba(features)

            if (prediction == 0): cancerPrediction = 'no'
            elif (prediction == 1): cancerPrediction = 'yes'
            if (prediction == 0): predictionProbability = str(probability[0][0]*100)+'%'
            elif (prediction == 1): predictionProbability = str(probability[0][1]*100)+'%'
            if (prediction == 0): recommendation = 'Safe to forgo screening'
            elif (prediction == 1): recommendation = 'Screening highly recommended'

            package = {
                'statusCode': 200,
                'results' : {
                    'prediction' : cancerPrediction,
                    'probability' : predictionProbability,
                    'recommendation' : recommendation
                }
            }

            package = json.dumps(package, indent=4)
            return package 
        
        else: return 'Payload keys incorrect'
    
    else: return 'Payload not of type JSON'
    
if __name__ == '__main__':
    application.run(debug=True) # deployment: remove "debug=True"