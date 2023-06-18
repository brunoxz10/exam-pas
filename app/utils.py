import pandas as pd
from pandas.api.types import CategoricalDtype
import pickle
import config


with open('../ml_dev/models/xgboost_categorical_not_calibrated.pickle', 'rb') as f:
    model = pickle.load(f)


# def preprocess_input_features(features: dict) -> pd.DataFrame:
#     new_sample = features
#     new_sample = pd.DataFrame([{col: new_sample.get(col) for col in config.FEATURES}])
#     new_sample = new_sample.fillna(0)
#     return new_sample


def preprocess_input_features(features: dict) -> pd.DataFrame:
    new_sample = features
    new_sample = pd.DataFrame([new_sample])
    cat_type = CategoricalDtype(categories=config.COURSE_NAMES)
    new_sample.course = new_sample.course.astype(cat_type)
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

    sample_approved = {
                    "escore_bruto_p1_etapa1": 6.034,
                    "escore_bruto_p2_etapa1": 64.65,
                    #"nota_redacao_etapa1": 9.733,
                    "escore_bruto_p1_etapa2": 3.845,
                    "escore_bruto_p2_etapa2": 63.826,
                    #"nota_redacao_etapa2": 9.933,
                    "escore_bruto_p1_etapa3": 7.14,
                    "escore_bruto_p2_etapa3": 76.636,
                    #"nota_redacao_etapa3": 9.931,
                    "pseudo_argumento_final": 70.36833333333334,
                    "min_flag": True,
                    "median_flag": True,
                    #"cotista": 0,
                    "cotas_negros_flag": 0,
                    #"publicas_flag": 0,
                    "publicas1_flag": 0,
                    "publicas2_flag": 0,
                    "publicas3_flag": 0,
                    "publicas4_flag": 0,
                    "publicas5_flag": 0,
                    "publicas6_flag": 0,
                    "publicas7_flag": 0,
                    "publicas8_flag": 0,
                    "course": "MEDICINA (BACHARELADO)"
                    }
                        
    predictions = predict_approval(sample_approved)
    print(predictions)