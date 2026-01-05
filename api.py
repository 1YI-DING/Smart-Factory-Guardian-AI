from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import joblib
import uvicorn

app = FastAPI(title="Equipment Health Monitoring API")


rf_model = joblib.load('rf_model.pkl')
iso_model = joblib.load('iso_model.pkl')


class SensorInput(BaseModel):
    vibration: float
    temperature: float
    pressure: float
    operating_hours: float
    vibration_mean: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vibration": 0.85,
                "temperature": 82.5,
                "pressure": 110.0,
                "operating_hours": 450.0,
                "vibration_mean": 0.82
            }
        }
    )


@app.post('/predict')
async def predict_failure(input: SensorInput):
    data = [[
        input.vibration,
        input.temperature,
        input.pressure,
        input.operating_hours,
        input.vibration_mean
    ]]

    failure = rf_model.predict(data)[0]
    prob = rf_model.predict_proba(data)[0][1]
    is_anomaly = iso_model.predict(data)[0] == -1

    return {
        'failure_prediction': 'High Risk' if failure == 1 else 'Normal',
        'failure_probability': f"{prob:.2%}",
        'anomaly_detected': bool(is_anomaly)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)