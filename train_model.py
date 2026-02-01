from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r"C:\Users\ADMIN\Desktop\Projects\HSE_VisionGuard\data_custom.yaml", epochs=100, imgsz=640)