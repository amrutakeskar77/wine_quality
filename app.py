from flask import Flask, render_template, request
import joblib


app = Flask(__name__)


obj = joblib.load('redwinequality.joblib')
model = obj['model']
cols = obj['columns']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():

    fixed_acidity = float(request.args.get('fixed_acidity'))
    volatile_acidity = float(request.args.get('volatile_acidity'))
    citric_acid = float(request.args.get('citric_acid'))
    residual_sugar = float(request.args.get('residual_sugar'))
    chlorides = float(request.args.get('chlorides'))
    free_sulfur_dioxide = float(request.args.get('free_sulfur_dioxide'))
    total_sulfur_dioxide = float(request.args.get('total_sulfur_dioxide'))
    density = float(request.args.get('density'))
    pH = float(request.args.get('pH'))
    sulphates = float(request.args.get('sulphates'))
    alcohol = float(request.args.get('alcohol'))


    Input = [[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]]


    out = model.predict(Input)

    prediction = float(out[0])


    return render_template(
        'index.html',
        prediction=prediction
    )


if __name__ == '__main__':
    app.run(debug=True)