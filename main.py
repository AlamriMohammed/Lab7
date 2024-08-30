from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

model = joblib.load('dbscan.joblib')
scaler = joblib.load('scaler1.joblib')
 
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
    return scaler.transform([feature_list])

@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.get("/try/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)  # Use predict instead of fit_predict
    return {"cluster": int(y_pred[0])}