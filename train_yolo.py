from ultralytics import YOLO
import time  # Added to measure time

model = YOLO("yolo11l.pt")

start_time = time.time()  # Start timing
model.train(data="data.yaml", imgsz=640, batch=8, epochs=100, workers=0, device=0)
end_time = time.time()  # End timing

print(f"Training completed in {(end_time - start_time) * 1000:.2f} ms")  # Print elapsed time in ms
