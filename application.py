from flask import Flask, request, render_template
import joblib

application = Flask(__name__)

model = joblib.load('XGBmodel.pkl')
scaler = joblib.load('scaler.pkl')

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

    features = scaler.transform([features])

    prediction_result = model.predict(features)
    prediction_probability = model.predict_proba(features)

    if (prediction_result == 0):
        return render_template('index.html', prediction = 'The model is '+str((prediction_probability[0][0])*100)+'% confident that you DO NOT have colon cancer. Please remember that this is not a diagnosis.')
    if (prediction_result == 1):
        return render_template('index.html', prediction = 'The model is '+str((prediction_probability[0][1])*100)+'% confident that you should get a screening for colon cancer. Please remember that this is not a diagnosis.')
    
if __name__ == '__main__':
    application.run(debug=True)