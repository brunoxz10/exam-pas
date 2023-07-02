import json
import pandas as pd
from flask import Flask, jsonify, request
from utils import predict_approval
import datetime
import config
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

df_scores = pd.read_parquet('../data/processed/scores_approvals_convocation_2020_2022.parquet')
df_scores = df_scores[config.RESULTS_INFO]

@app.route('/filter', methods=['GET']) 
def filter_dataframe():
    numero_inscricao = request.args.get('numero_inscricao')  # Get 'name' parameter from the request

    if numero_inscricao:
        filtered_df = df_scores[df_scores['numero_inscricao'] == numero_inscricao]  # Filter DataFrame based on 'name'
        result = filtered_df.to_dict('records')
    else:
        result = df_scores.to_dict('records')  # Return the entire DataFrame if 'name' parameter is not provided

    return jsonify(result)


@app.post('/predict') 
def predict():
    
    # this is a python dictionary 
    data = request.json
           
    approval_prediction = predict_approval(data)
        
    try:
        result = jsonify({'metadata': {"timestamp": str(datetime.datetime.now())},
                          'prediction': {"probability": approval_prediction}})
    
    except TypeError as e:
        return jsonify({'error': str(e)})
        
    return result
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
