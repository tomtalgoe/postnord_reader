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

# Initialize EasyOCR
reader = easyocr.Reader(["en"])


def logline(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}(postnord_ocr) - {message}")


# Region of intrest/ROI with Yolo11 model
model = YOLO("runs/detect/train7/weights/best.pt")


# doing
def roi_ocr(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]  # Get first result
    if results.boxes is None or results.boxes.xyxy.shape[0] == 0:
        logline("No bounding boxes detected")
        return None  # or return image if you want to fall back

    # Get the box with highest confidence
    best_box = results.boxes.xyxy[results.boxes.conf.argmax()].cpu().numpy().astype(int)
    x1, y1, x2, y2 = best_box
    roi = image[y1:y2, x1:x2]  # Crop the image to the detected box

    return roi, (x1, y1, x2, y2)


def processed_image(image_aroi):
    image = image_aroi

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contrast = cv2.convertScaleAbs(gray, alpha=1, beta=10)

    blur = cv2.GaussianBlur(contrast, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return blur


def extract_text(image):
    # OCR for first three characters (letters only)
    letters = reader.readtext(image, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    letters_text = "".join(letters).strip().upper()

    # OCR for last three characters (numbers only)
    numbers = reader.readtext(image, detail=0, allowlist="0123456789")
    numbers_text = "".join(numbers).strip()

    # Combine results
    extracted_text = f"{letters_text[:3]}{numbers_text[-3:]}"
    logline(
        f"Extracted: {extracted_text} (Letters: {letters_text}, Numbers: {numbers_text})"
    )
    return extracted_text


def process_dataset(dataset_path):

    results = []
    image_files = [
        f for f in os.listdir(dataset_path) if f.endswith((".png", ".jpg", ".jpeg"))
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
    df_results.to_csv("ocr_results.csv", index=False)

    print(df_results)


def process_image(image_path):
    image_aroi, bbox = roi_ocr(image_path)
    if image_aroi is None:
        raise ValueError("No ROI found in image. Skipping processing.")

    # rotated = cv2.rotate(image_aroi, cv2.ROTATE_90_CLOCKWISE)

    processed_img = processed_image(image_aroi)
    extracted_text = extract_text(processed_img)

    return processed_img, extracted_text, image_aroi, bbox
