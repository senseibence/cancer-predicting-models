from flask import Flask, request, jsonify, render_template
import joblib
import numpy

app = Flask(__name__)
model = joblib.load('XGBoost_DT.pkl')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/index.html")
def index():
    return render_template('index.html')

@app.route("/pages/about.html")
def about():
    return render_template('/pages/about.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = request.form.to_dict()
    features = list(features.values())
    features = list(map(int, features))
  
    prediction = model.predict([features])

    result = 'Failed'
    if (prediction == 0): result = 'No'
    if (prediction == 1): result = 'Yes'

    return render_template('index.html', prediction = 'Colon Cancer Prediction: '+result)

if __name__ == '__main__':
     app.run(debug=True)