import json
import pandas as pd
from flask import Flask, jsonify, request
from utils import predict_approval
import config

app = Flask(__name__)

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


@app.post('/predict') 
def predict():
    
    # this is a dictionary 
    data = request.json
    
    input_features_valid = all([col in config.FEATURES for col in list(data.keys())])
    
    if not input_features_valid:
        return jsonify({'error':'Invalid input'})
        
    approval_prediction = predict_approval(data)
        
    try:
        result = jsonify({'Output':f'A sua probabilidade de aprovação é {approval_prediction}'})
    
    except TypeError as e:
        return jsonify({'error':str(e)})
        
    return result
    
    # else:
    #     return jsonify({'error':'Invalid input'})

    #return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
