import requests
import joblib
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

#TODO: Add inference on whole dataset in order to compute back recall and verify that there is no loss during the deployment
def load_dataset_for_inference(dataset_path):
    df = pd.read_csv("../train/nba_logreg.csv")

    #Fill nan by 0
    df.fillna(0.0, inplace=True)

    # Remove duplicated rows
    df.drop_duplicates(inplace=True)

    df.drop(columns = ['PTS' , 'DREB', 'FTM', 'FGA' ,'3PA'] , axis = 1, inplace = True)
    print(df.head())
    # Extract fetaures and label
    df_vals = df.drop(['TARGET_5Yrs','Name'],axis=1)
    labels = df['TARGET_5Yrs'].values
    return df_vals, labels


def get_player_characteristic_from_json(input_filename):
    f = open(input_filename)
    data = json.load(f)
    inputs_list = []
    print(data)
    for k, v in data.items():
        inputs_list.append(v)
    return np.asarray([inputs_list])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='player_characteristic.json')   
    args = parser.parse_args()
    return args

def infer_one_player(X_scaled):
    print(X_scaled.shape)
    print(X_scaled.tolist())
    five_year = -1
    inference_request = {
        "inputs": [
            {
            "name": "predict",
            "shape": X_scaled.shape,
            "datatype": "FP64",
            "data": X_scaled.tolist()
            }
        ]
    }
    endpoint = "http://0.0.0.0:8080/v2/models/nba_rookie_autoMl_scorer_recall/versions/v0.1.0/infer"
    response = requests.post(endpoint, json=inference_request)
    print(response)

    if response.status_code == 200:

        for r in response.json()['outputs']:
            res = r['data'][0]   
            print(res)
            if res == 1:  
                five_year = 1  
                print('This player will stay in NBA for at least 5 years')
            else:
                five_year = 0
                print('This player will not stay in NBA for at least 5 years')
    else:
        print('SOMETHING WENT WRONG')
        print('RESPONSE STATUS CODE : ', response.status_code)
    return five_year

def main():
    args = parse_args()

    scaler = joblib.load('nba_rookie_mmscaler.save')

    'To verify that deployment didn t decrease performance'
    df_vals, labels = load_dataset_for_inference('../train/nba_logreg.csv')
    tp, fp = 0, 0
    X_scaled = scaler.transform(df_vals)
    predicted_labels = []
    for player_characteristic, label in zip(X_scaled, labels):
        #print('player_characteristic, label : ', player_characteristic, label)
        print('start inference')
        five_year = infer_one_player(np.expand_dims(np.asarray(player_characteristic), axis = 0))
        print(five_year, label)
        predicted_labels.append(five_year)

    recall = recall_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    f1= f1_score(labels, predicted_labels)
    confusion_mat = confusion_matrix(labels,predicted_labels)
    print("Average Final confusion matrix: ", confusion_mat)
    print("Average Final precision: ", precision)
    print("Average Final f1: ", f1)
    print("Average Final recall: ", recall)

    '''    
    player_characteristic = get_player_characteristic_from_json(args.input)

    X_scaled = scaler.transform(player_characteristic)
    
    five_year = infer_one_player(X_scaled)
    '''

if __name__ == '__main__':
    main()