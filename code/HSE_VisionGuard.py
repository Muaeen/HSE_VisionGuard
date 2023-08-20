from ultralytics import YOLO
model = YOLO("best.pt")

# model.predict(source=1,show=True,save=False,conf=0.5)

while True:
    results = model(source=0, show=True, save=False, conf=0.5, stream=True)
    
    for result in results:
        # detection
        x = result.boxes.xyxy   # box with xyxy format, (N, 4)
        result.boxes.xywh   # box with xywh format, (N, 4)
        result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        result.boxes.conf   # confidence score, (N, 1)
        result.boxes.cls    # cls, (N, 1)
    
        # classification
        result.probs     # cls prob, (num_class, )
        
        print(x)
