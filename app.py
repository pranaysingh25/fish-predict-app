from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('fish_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])

        # Predict species
        features = np.array([[weight, length1, length2, length3, height, width]])
        species = model.predict(features)[0]

        return redirect(url_for('result', species=species))

@app.route('/result')
def result():
    species = request.args.get('species')
    return render_template('result.html', species=species)

if __name__ == '__main__':
    app.run(debug=True)
