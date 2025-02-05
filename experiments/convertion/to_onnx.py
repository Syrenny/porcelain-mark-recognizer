import torch
import onnx
from torchvision import models


def convert_pth_to_onnx(model_pth_path: str, onnx_model_path: str):
    # Создаем модель (например, ResNet-18)
    # Важно указать, что мы не используем предобученные веса
    model = models.resnet18(pretrained=False)

    # Загружаем веса из .pth файла
    model.load_state_dict(torch.load(model_pth_path))

    # Переводим модель в режим инференса
    model.eval()

    # Создаем фиктивное изображение (например, 3 канала, 224x224 пикселя)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Экспортируем модель в формат ONNX
    torch.onnx.export(model, dummy_input, onnx_model_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=11)

    print(f"Модель успешно сохранена в {onnx_model_path}")


# Пример использования:
# Путь к вашей модели .pth
model_pth_path = "/home/asphodel/comptech2025-porcelain/porcelain-mark-recognizer/models/search/resnet18-f37072fd.pth"
# Путь, где будет сохранена модель .onnx
onnx_model_path = "/home/asphodel/comptech2025-porcelain/porcelain-mark-recognizer/models/search/resnet18.onnx"

convert_pth_to_onnx(model_pth_path, onnx_model_path)
