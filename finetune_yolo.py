from ultralytics import YOLO

# Path to YOLOv11 repo
model = YOLO("runs/detect/train7/weights/best.pt")


# Train the model
model.train(data="data.yaml", epochs=20, imgsz=640, batch=16, device=0)
