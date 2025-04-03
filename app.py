from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import sklearn
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = joblib.load('lending_intelligence_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:

            no_of_dependents = float(request.form['no_of_dependents'])
            education = request.form['education']
            self_employed = request.form['self_employed']
            income_annum = float(request.form['income_annum'])
            loan_amount = float(request.form['loan_amount'])
            loan_term = float(request.form['loan_term'])
            cibil_score = float(request.form['cibil_score'])
            residential_assets_value = float(request.form['residential_assets_value'])
            commercial_assets_value = float(request.form['commercial_assets_value'])
            luxury_assets_value = float(request.form['luxury_assets_value'])
            bank_asset_value = float(request.form['bank_asset_value'])

            education_encoded = 1 if education == "Graduate" else 0
            self_employed_encoded = 1 if self_employed == "Yes" else 0

            columns = [
                ' no_of_dependents', ' education', ' self_employed', ' income_annum',
                ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value',
                ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value'
            ]

            input_data = pd.DataFrame(
                [[no_of_dependents, education_encoded, self_employed_encoded,
                  income_annum, loan_amount, loan_term, cibil_score,
                  residential_assets_value, commercial_assets_value,
                  luxury_assets_value, bank_asset_value]],
                columns=columns
            )

            logger.debug(f"Input data: {input_data.to_dict()}")

            raw_prediction = model.predict(input_data)[0]
            logger.debug(f"Raw prediction: {raw_prediction}")

            if isinstance(raw_prediction, (np.ndarray, list)):

                prediction = {
                    'loan_approved': bool(raw_prediction[0]),
                    'approval_probability': float(raw_prediction[1]),
                    'risk_score': float(raw_prediction[2])
                }
            elif hasattr(model, 'predict_proba'):

                approval_prob = model.predict_proba(input_data)[0][1]
                prediction = {
                    'loan_approved': bool(raw_prediction),
                    'approval_probability': approval_prob,
                    'risk_score': None  # Adjust if risk_score is calculated separately
                }
            else:

                prediction = {
                    'loan_approved': bool(raw_prediction),
                    'approval_probability': None,
                    'risk_score': None
                }

            logger.debug(f"Formatted prediction: {prediction}")

        except ValueError as ve:
            logger.error(f"ValueError: {str(ve)}")
            prediction = {'error': f"Invalid input - please enter numeric values where required ({str(ve)})"}
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            prediction = {'error': f"Something went wrong ({str(e)})"}

    return render_template('index.html', prediction=prediction)


if __name__ == '_main_':
    app.run(debug=True)
