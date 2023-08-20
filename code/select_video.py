from ultralytics import YOLO

model = YOLO("best.pt")

sourcr = '/Users/ahmedalmaqbali/Desktop/HSE_VisionGuard_2/code/Health and Safety Consultant (Episode 140).mp4'

# Parameters to skip frames and reduce display frequency
skip_frames = 5  # process every 5th frame
display_frequency = 10  # update the display every 10 frames
frame_counter = 0

results = model(source=sourcr, show=False, save=False, conf=0.5, stream=True)

for result in results:
    frame_counter += 1

    # Only process every `skip_frames` frames
    if frame_counter % skip_frames != 0:
        continue

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

    # Update the display less frequently
    if frame_counter % display_frequency == 0:
        result.show()
