from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import uvicorn

app = FastAPI()

# Заглушка для нейросетевого алгоритма


def mock_predict(image_bytes: bytes) -> List[dict]:
    # Здесь должен быть вызов нейросетевой модели для получения предсказаний
    return [
        {"name": "Марка A", "description": "Описание марки A",
            "image_url": "https://via.placeholder.com/150"},
        {"name": "Марка B", "description": "Описание марки B",
            "image_url": "https://via.placeholder.com/150"},
        {"name": "Марка C", "description": "Описание марки C",
            "image_url": "https://via.placeholder.com/150"}
    ]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predictions = mock_predict(image_bytes)
    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
