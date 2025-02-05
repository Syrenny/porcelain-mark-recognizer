import cv2
import numpy as np
from ultralytics import YOLO

from environment import settings
    

class Detector:
    """
    Класс для детекции объектов с помощью YOLO и обрезки найденных боксов.
    """
    def __init__(self):
        """
        Инициализация YOLO модели.

        :param model_path: Путь к весам YOLO модели.
        :param conf_threshold: Порог уверенности для фильтрации детекций.
        """
        self.model = YOLO(settings.yolo_model_path)

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Обрабатывает изображение и возвращает список вырезанных объектов (кропов).

        :param image_path: Путь к изображению.
        :return: Список вырезанных изображений (numpy массивов).
        """
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb)

        crops = []
        for box in results[0].boxes:
            if box.conf[0] < settings.yolo_conf_threshold:
                continue  # Пропускаем боксы с низкой уверенностью

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            cropped_image = image[y1:y2, x1:x2]
            crops.append(cropped_image)

        return crops
