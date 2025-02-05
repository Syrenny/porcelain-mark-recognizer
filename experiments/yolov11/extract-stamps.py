import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Инициализация модели
model = YOLO("./runs/detect/train16/weights/best.pt")

def process_image(image_path, output_dir, base_filename, conf_threshold=0.5):
    """
    Применение YOLO к изображению, вырезание найденных боксов, сохранение их и
    создание отладочных изображений с разметкой.
    """
    results = model(image_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for idx, box in enumerate(results[0].boxes):  # Получаем боксы
        if box.conf[0] < conf_threshold:
            continue  # Пропускаем боксы с низкой уверенностью

        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты одного бокса

        # Проверка на выход за границы изображения
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        cropped_image = image[y1:y2, x1:x2]

        # Сохраняем вырезанное изображение
        crop_output_path = os.path.join(output_dir, f"{base_filename}_crop_{idx + 1}.png")
        cv2.imwrite(crop_output_path, cropped_image)

def process_directory(input_dir, output_dir):
    """
    Рекурсивная обработка изображений во всех подпапках с использованием rglob.
    """
    input_path = Path(input_dir)
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    
    for ext in image_extensions:
        for file_path in input_path.rglob(ext):
            relative_path = file_path.parent.relative_to(input_path)
            output_class_dir = Path(output_dir) / relative_path
            output_class_dir.mkdir(parents=True, exist_ok=True)
            base_filename = file_path.stem
            process_image(str(file_path), str(output_class_dir), base_filename)

if __name__ == "__main__":
    input_directory = "./data/raw/stamps"      # Папка с исходными изображениями
    output_directory = "./data/transformed/cropped_dataset"      # Папка для сохранения результатов

    process_directory(input_directory, output_directory)
