from ultralytics import YOLO
import os

# Load a model
model = YOLO(r"C:\Users\ADMIN\Desktop\Projects\HSE_VisionGuard\runs\detect\train2\weights\best.pt")

# Get video path from user or use default
video_path = r"C:\Users\ADMIN\Desktop\Projects\HSE_VisionGuard\Health and Safety Consultant (Episode 140).mp4"
if not video_path:
    from tkinter import filedialog
    video_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
    )

if not video_path or not os.path.exists(video_path):
    print("No valid video file selected.")
    exit()

# Run inference on video and save annotated video
results = model.predict(
    source=video_path,
    save=True,  # Save annotated video
    conf=0.3,   # Confidence threshold
    save_txt=False,  # Don't save text files
    save_conf=True,  # Include confidence in labels
    project="runs/detect",  # Save location
    name="video_inference"  # Subfolder name
)

print(f"\nAnnotated video saved to: {results[0].save_dir}")
print("Processing complete!")