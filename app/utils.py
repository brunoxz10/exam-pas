import pandas as pd
import pickle
import config


with open('../ml_dev/models/stratified_kfold_classweight15_tuned_model.pickle', 'rb') as f:
    model = pickle.load(f)


def preprocess_input_features(features: dict) -> pd.DataFrame:
    new_sample = features
    new_sample = pd.DataFrame([{col: new_sample.get(col) for col in config.FEATURES}])
    new_sample = new_sample.fillna(0)
    return new_sample


def predict_approval(new_data: pd.DataFrame):
    
    new_data = preprocess_input_features(new_data)
    
    # Predict diabetes
    predictions = model.predict_proba(new_data)
    approval_prob = predictions[0][1]

    #pred_to_label = {0: 'Negative', 1: 'Positive'}

    # Make a list of predictions
    #data = []
    #for t, pred in zip(new_data, predictions):
    #    data.append({'prediction': pred[0]})

    return f'{round(approval_prob*100, ndigits=1)}%'


if __name__=="__main__":

    sample_not_approved = {"escore_bruto_p1_etapa1": 5.172,
                           "escore_bruto_p2_etapa1": 14.653,
                           "nota_redacao_etapa1": 6.947,
                           "escore_bruto_p1_etapa2": 3.845,
                           "escore_bruto_p2_etapa2": 19.994,
                           "nota_redacao_etapa2": 7.222,
                           "escore_bruto_p1_etapa3": 4.998,
                           "escore_bruto_p2_etapa3": 16.66,
                           "nota_redacao_etapa3": 8.06,
                           #"argumento_final": -25.699,
                           "2.1.1 CAMPUS  DARCY RIBEIRO – DIURNO  ADMINISTRAÇÃO (BACHARELADO)": 1.0,               
                           "cotista": 0.0,
                           "cotas_negros_flag": 0.0,
                           "publicas1_flag": 0.0,
                           "publicas2_flag": 0.0,
                           "publicas3_flag": 0.0,
                           "publicas4_flag": 0.0,
                           "publicas5_flag": 0.0,
                           "publicas6_flag": 0.0,
                           "publicas7_flag": 0.0,
                           "publicas8_flag": 0.0}
    
    predictions = predict_approval(sample_not_approved)
    print(predictions)