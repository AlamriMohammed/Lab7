from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()
@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"

@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}


# url = "http://localhost:8000/me"
# data = {
# "Year": 2020,
# "goals": 2.5,
# "current_value": 15000,
# "Type": "Accent",
# "Make": "Hyundai",
# "Options": "Full"
# }
# response = requests.post(url, json=data)
# print(response.json())


import joblib
# model = joblib.load('knn_model.joblib')
model = joblib.load("dbscan.joblib")
scaler = joblib.load("scaler.joblib")

from pydantic import BaseModel
# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    # highest_value: int
    current_value: int
    goals: float
 

def preprocessing(input_features: InputFeatures):
    dict_f = {
    # 'highest_value': input_features.highest_value
        'current_value': input_features.current_value,
        'goals': input_features.goals

    }
    feature_list = [dict_f[key] for key in sorted(dict_f)]
    return scaler.transform([list(dict_f.values())])

    # return features_list.value()

@app.get("/predict")
def predict(input_features: InputFeatures): 
    return preprocessing(input_features)

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}