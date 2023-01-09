from flask import Flask, request, render_template
import joblib

application = Flask(__name__)

file = open("model.pkl", "rb")
model = joblib.load(file)

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
    
    prediction = model.predict([features])

    result = 'Failed'
    if (prediction == 0): result = 'No'
    if (prediction == 1): result = 'Yes'

    return render_template('index.html', prediction = 'Colon Cancer Prediction: '+result)

if __name__ == '__main__':
    application.run()