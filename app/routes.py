from flask import  request, jsonify
from app.eda.eda_funcs import load_model, data_scaling, preprocessing
from app import app
import pandas as pd


def load_data(data):
    col_dict = {'accx': 'gFx',
                'accy': 'gFy',
                'accz': 'gFz',
                'linear_accX': 'ax',
                'linear_accY': 'ay',
                'linear_accZ': 'az',
                'gyrox': 'wx',
                'gyroy': 'wy',
                'gyroz': 'wz',
                'magnetox': 'Bx',
                'magnetoy': 'By',
                'magnetoz': 'Bz'}
    dataset = preprocessing(pd.DataFrame(data).rename(columns=col_dict))
    return data_scaling(dataset)

## Load Model
model = load_model()

@app.route('/prediction', methods=['POST', 'GET'])
def home():
    # get the data
    input_data = request.json
    if not input_data:
        return jsonify({"result":"Incorrect input data", "error": True})
    # preprocess the data
    dataset = load_data(input_data)
    prediction = model.predict(dataset)
    if len(prediction) != 1:
        return jsonify({"result":"Error in Predicting Value", "error": True})

    return jsonify({"result":prediction[0], "message": "success"})
    


