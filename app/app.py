import json
import pandas as pd
from flask import Flask, jsonify, request
#from utilities import predict_diabetes

app = Flask(__name__)

# covariables = ['Pregnancies',
#                'Glucose',
#                'BloodPressure',
#                'SkinThickness',
#                'Insulin',
#                'BMI',
#                'DiabetesPedigreeFunction',
#                'Age']

df_scores = pd.read_parquet('../data/scores.parquet')

@app.route('/filter', methods=['GET']) 
def filter_dataframe():
    numero_inscricao = request.args.get('numero_inscricao')  # Get 'name' parameter from the request

    if numero_inscricao:
        filtered_df = df_scores[df_scores['numero_inscricao'] == numero_inscricao]  # Filter DataFrame based on 'name'
        result = filtered_df.to_dict('records')
    else:
        result = df_scores.to_dict('records')  # Return the entire DataFrame if 'name' parameter is not provided

    return jsonify(result)


# @app.route('/filter', methods=['GET']) 
# def predict():
#     data = request.json
#     print(data)

#     try:
#         data = pd.DataFrame(data)
#     except ValueError:
#         data = pd.DataFrame([data])

#     sample = data.values
#     return jsonify(sample)

#     # if list(data.columns) == covariables:
#     #     try:
#     #         sample = data.values
#     #     except KeyError:
#     #         return jsonify({'error':'Invalid input'})
        
#     #     predictions = predict_diabetes(sample)
        
#     #     try:
#     #         result = jsonify(predictions)
#     #     except TypeError as e:
#     #         return jsonify({'error':str(e)})
        
#     #     return result
    
#     # else:
#     #     return jsonify({'error':'Invalid input'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)