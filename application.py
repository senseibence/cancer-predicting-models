from flask import Flask, request, render_template, make_response, redirect
import joblib
import json
# import keras

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

# Neuralmodel = keras.models.load_model('Neuralmodel.h5')
# Neuralscaler = joblib.load('Neuralscaler.pkl')

def calculateBMI(features):
    feet = features[1]
    inches = features[2]
    converted_inches = inches / 12
    feet += converted_inches

    meters = feet / 3.281
    if (meters == 0): return 27.282130617023647 # this is mean of BMI column; division by zero if unchecked

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

@application.route("/about.html")
def about():
    return render_template('/about.html')

@application.route("/api.html")
def api():
    return render_template('/api.html')

@application.route("/questionnaires/colon.html")
def colon():
    cookie = request.cookies.get('ToS_agreed')
    if (cookie == 'true'): return render_template('/questionnaires/colon.html')
    else: return redirect('/legal/agreement.html')

@application.route("/questionnaires/pancreatic.html")
def pancreatic():
    return render_template('/questionnaires/pancreatic.html')

@application.route("/questionnaires/lung.html")
def lung():
    return render_template('/questionnaires/lung.html')

@application.route("/legal/disclaimer.html")
def disclaimer():
    return render_template('/legal/disclaimer.html')

@application.route("/legal/terms-of-service.html")
def terms_of_service():
    return render_template('/legal/terms-of-service.html')

@application.route("/legal/privacy-policy.html")
def privacy_policy():
    return render_template('/legal/privacy-policy.html')

@application.route("/legal/cookie-policy.html")
def cookie_policy():
    return render_template('/legal/cookie-policy.html')

@application.route("/legal/agreement.html")
def agreement():
    cookie = request.cookies.get('ToS_agreed')
    if (cookie == 'true'): return redirect('/questionnaires/colon.html')
    else: return render_template('/legal/agreement.html')

@application.route('/setcookie', methods=['POST'])
def setcookie():
    answer = request.form.get('agreement')
    if (answer == '1'):
        resp = make_response(redirect('/questionnaires/colon.html'))
        resp.set_cookie('ToS_agreed', 'true')
        return resp
    else:
        resp = make_response(redirect('/legal/agreement.html'))
        resp.set_cookie('ToS_agreed', 'false')
        return resp

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
                except: return 'Payload values not convertible to type int or float'
                features.append(float(value))
                
            bmi = calculateBMI(features)
            del features[1:4]
            features.insert(14, bmi)

            # ML Models
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
                'statusCode' : 200,
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
    application.run() # deployment: remove debug=True