from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

from backend.detector import Detector
from backend.search import SearchEngine, IndexData


app = FastAPI()

# Инициализация классов вместе с загрузкой нейросетей в память
yolo = Detector()
search_engine = SearchEngine()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> list[IndexData]:
    np_image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    # Выход - набор кропов с изображения
    crops: list = yolo(image)
    # Выбираем первый кроп и ищем марки
    # Ожидаем, что нам вернется json-объект с
    # картинкой марки, производителем и метадатой
    result = search_engine(crops[0])
    return result
