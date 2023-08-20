import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

# Open the video source
cap = cv2.VideoCapture('/Users/ahmedalmaqbali/Desktop/HSE_VisionGuard_2/code/Health and Safety Consultant (Episode 140).mp4')

# Set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Define class names and colors
class_names = ["helmet", "vest"]
class_colors = [(0, 255, 0), (0, 0, 255)]  # Green for helmet, Red for vest

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get detections from YOLO
    results = model(frame, conf=0.3)
    
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[int(cls)]}: {conf:.2f}"
            
            # Fetch class color
            color = class_colors[int(cls)]
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Write the frame to the output video
    out.write(frame)

    # Optionally show the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video source and writer
cap.release()
out.release()
cv2.destroyAllWindows()

print("successful")

