from fastapi import FastAPI, Request
import joblib
import pandas as pd
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
# 저장한 model, scaler 불러오기
model = joblib.load('./model.pkl')
scaler = joblib.load('./scaler.pkl')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
async def index():
    return {
        'message': '아이리스 종 예측'
    }

@app.post('/predict')
async def predict(iris_request: IrisRequest):
    data = iris_request.dict()
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    species = ['Setosa', 'Versicolour', 'Virginica']
    result = species[prediction[0]]
    return {
        'message': f'예측된 아이리스 종은 {result}입니다.'
    }

# FastAPI 서버 실행을 위한 코드
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
