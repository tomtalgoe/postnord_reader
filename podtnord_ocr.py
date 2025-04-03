from csv import reader
import cv2
import easyocr
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
import traceback
import psutil  # Add this import to monitor memory usage

# Initialize EasyOCR
reader = easyocr.Reader(["en"])


def logline(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}(postnord_ocr) - {message}")


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logline(f"Memory usage: RSS={memory_info.rss / (1024 * 1024):.2f} MB, VMS={memory_info.vms / (1024 * 1024):.2f} MB")


# Normalize YOLO model path
model_path = os.path.join("runs", "detect", "train7", "weights", "best.pt")
if not os.path.exists(model_path):
    logline(f"YOLO model path does not exist: {model_path}")
model = YOLO(model_path)


def roi_ocr(image_path):
    try:
        log_memory_usage()  # Log memory usage before processing
        # Normalize image path
        image_path = os.path.normpath(image_path)
        logline(f"Processing image: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            logline(f"Failed to read image: {image_path}")
            return None, None

        # Ensure YOLO model runs synchronously
        results = model(image)[0]  # Get first result
        if results.boxes is None or results.boxes.xyxy.shape[0] == 0:
            logline("No bounding boxes detected")
            return None, None

        # Get the box with highest confidence
        best_box = results.boxes.xyxy[results.boxes.conf.argmax()].cpu().numpy().astype(int)
        x1, y1, x2, y2 = best_box
        roi = image[y1:y2, x1:x2]  # Crop the image to the detected box

        return roi, (x1, y1, x2, y2)
    except Exception as e:
        logline(f"Error in roi_ocr: {e}")
        traceback.print_exc()
        return None, None


def processed_image(image_aroi):
    try:
        log_memory_usage()  # Log memory usage before processing
        image = image_aroi

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logline("Converted to grayscale")

        contrast = cv2.convertScaleAbs(gray, alpha=1, beta=10)

        blur = cv2.GaussianBlur(contrast, (3, 3), 0)
        logline("Applied Gaussian blur")

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        logline("Applied adaptive thresholding, blur: {}".format(blur.shape))

        return blur
    except Exception as e:
        logline(f"Error processing image: {e}")
        traceback.print_exc()
        return null


def extract_text(image):
    try:
        # Ensure EasyOCR runs synchronously
        letters = reader.readtext(image, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        letters_text = "".join(letters).strip().upper()

        numbers = reader.readtext(image, detail=0, allowlist="0123456789")
        numbers_text = "".join(numbers).strip()

        extracted_text = f"{letters_text[:3]}{numbers_text[-3:]}"
        logline(f"Extracted: {extracted_text} (Letters: {letters_text}, Numbers: {numbers_text})")
        return extracted_text
    except Exception as e:
        logline(f"Error in extract_text: {e}")
        traceback.print_exc()
        return ""


def process_dataset(dataset_path):
    try:
        # Normalize dataset path
        dataset_path = os.path.normpath(dataset_path)
        logline(f"Processing dataset: {dataset_path}")

        results = []
        image_files = [
            f for f in os.listdir(dataset_path) if f.endswith((".png", ".jpeg"))
        ]

        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)

            # Process Image
            processed_images = processed_image(image_path)

            # OCR on each version
            ocr_results = {
                version: extract_text(img) for version, img in processed_images.items()
            }

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)

        # Save results to CSV
        output_csv = os.path.join(dataset_path, "ocr_results.csv")
        df_results.to_csv(output_csv, index=False)
        logline(f"Results saved to: {output_csv}")

        print(df_results)
    except Exception as e:
        logline(f"Error in process_dataset: {e}")
        traceback.print_exc()


def process_image(image_path):
    try:
        log_memory_usage()  # Log memory usage before processing
        image_aroi, bbox = roi_ocr(image_path)
        if image_aroi is None:
            logline("No ROI found in image. Skipping processing.")
            return None, None, None, None

        processed_img = processed_image(image_aroi)
        if processed_img is None:
            logline("Processed image is None. Skipping further processing.")
            return None, None, image_aroi, bbox

        extracted_text = extract_text(processed_img)

        return processed_img, extracted_text, image_aroi, bbox
    except Exception as e:
        logline(f"Error processing image: {e}")
        traceback.print_exc()
        return None, None, None, None
    finally:
        logline("Finished processing image.")

