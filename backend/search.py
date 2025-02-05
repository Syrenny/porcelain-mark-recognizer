from pathlib import Path
import logging
import json

import numpy as np
import cv2
from scipy.spatial.distance import cosine
from enum import Enum
from torchvision import transforms
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm
from pydantic import BaseModel

from environment import settings


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class SiameseNetwork:
    def __init__(self):
        # Загружаем модель ONNX
        logger.info("Загружаем модель ONNX для сиамской сети...")

        self.session = ort.InferenceSession(settings.search_model_path)
        logger.info(f"Модель загружена из {settings.search_model_path}")

        # Преобразования для подготовки изображения
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        # Преобразуем изображение в формат, подходящий для модели
        image_pil = Image.fromarray(image)
        image_tensor = self.transform(image_pil).unsqueeze(
            0)  # Добавляем размерность батча
        image_tensor = image_tensor.numpy()  # Конвертируем в numpy

        # Выполняем инференс
        inputs = {self.session.get_inputs()[0].name: image_tensor}
        output = self.session.run(None, inputs)

        # Результат инференса - это эмбеддинг (вектор)
        return output[0].squeeze()  # Убираем размерность батча


class Fields(str, Enum):
    producer="producer"
    embedding="embeddings"
    path="path"


# Структура данных для хранения каждого индекса
class IndexData(BaseModel):
    producer: str
    embedding: list[float]
    path: Path

    # Метод для преобразования в JSON
    def to_json(self) -> str:
        return json.dumps({
            "producer": self.producer,
            "embedding": str(self.embedding),
            "path": self.path.as_posix()
        }, default=str)


# Проводим индексацию при инициализации класса
# Хочется, чтобы в list[dict] хранились данные индекса:
# путь до картинки, эмбеддинг, производитель, метадата
class SearchEngine:
    def __init__(self):
        logger.info("Инициализируем поисковую систему и загружаем модель...")
        self.siamese_net = SiameseNetwork()
        self.index_json: list[IndexData] = []
        self._index_images()

    def add_image_to_index(self, image_path: str, cls: str):
        image = read_image(image_path)
        embedding = self.siamese_net.get_embedding(image)
        self.index_json.append(IndexData(
            producer=cls,
            embedding=embedding,
            path=image_path.as_posix()
        ))

    def _index_images(self):
        images_dir = Path(settings.search_images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Директория {images_dir} не найдена.")

        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for image_path in tqdm(images_dir.rglob(ext), desc="Индексируем изображения"):
                self.add_image_to_index(
                    image_path=image_path,
                    cls=str(Path(image_path).parent)
                )

    def __call__(self, image: np.ndarray) -> list[IndexData]:
        """
        Выполняет поиск похожих изображений по эмбеддингу запроса,
        используя косинусное сходство.
        """
        # Получаем эмбеддинг запроса
        query_embedding = self.siamese_net.get_embedding(image)

        # Массив для хранения сходства (или расстояния) между запросом и каждым индексированным изображением
        similarity_scores = np.array(
            [1 - cosine(query_embedding, entry.embedding) for entry in self.index_json])

        # Сортируем индексы по сходству (по убыванию) и выбираем топ-K
        top_k_indices = np.argsort(similarity_scores)[
            ::-1][:settings.search_top_k]

        return np.array(self.index_json)[top_k_indices]
