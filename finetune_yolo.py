from ultralytics import YOLO
 
def main():
    model = YOLO("yolo11l.pt")  # Or your own weights
    model.train(data="data.yaml", epochs=20, imgsz=640, batch=16, device=0)
 
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional for scripts turned into .exe
    main()