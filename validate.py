from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\ADMIN\Desktop\Projects\HSE_VisionGuard\yolo26n.pt")  # load an official model
model = YOLO(r"C:\Users\ADMIN\Desktop\Projects\HSE_VisionGuard\runs\detect\train2\weights\best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list containing mAP50-95 for each category