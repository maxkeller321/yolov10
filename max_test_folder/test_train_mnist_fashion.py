from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='fashion-mnist', epochs=100, imgsz=28)