import comet_ml
from ultralytics import YOLO


comet_ml.login(project_name="porcelain-marks-detection")

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="./data/dataset.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model