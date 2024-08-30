from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
model = joblib.load('dbscan.joblib')
scaler = joblib.load('scaler.joblib')

app = FastAPI()

class InputFeatures(BaseModel):
    current_value: int
    goals: float

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'current_value': input_features.current_value,
        'goals': input_features.goals
    }
    feature_list = [dict_f[key] for key in sorted(dict_f)]
    if isinstance(scaler, np.ndarray):
        raise ValueError("The loaded object is a numpy array, not a scaler.")

    return scaler.transform([feature_list])

@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.fit_predict(data)
    return {"cluster": int(y_pred[0])}
