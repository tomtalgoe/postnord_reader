from csv import reader
import cv2
import easyocr
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# Initialize EasyOCR
reader = easyocr.Reader(["en"])


def processed_image(image_path):
    image = cv2.imread(image_path)

    # 1️⃣ Original
    original = image.copy()

    # 2️⃣ Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Contrast Boost
    contrast = cv2.convertScaleAbs(gray, alpha=1, beta=10)

    # 4️⃣ Blur (Noise Reduction)
    blur = cv2.GaussianBlur(contrast, (3, 3), 0)

    # 5️⃣ Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return thresh


def extract_text(image):
    # OCR for first three characters (letters only)
    letters = reader.readtext(image, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    letters_text = "".join(letters).strip().upper()

    # OCR for last three characters (numbers only)
    numbers = reader.readtext(image, detail=0, allowlist="0123456789")
    numbers_text = "".join(numbers).strip()

    # Combine results
    extracted_text = f"{letters_text[:3]}{numbers_text[-3:]}"

    # Validate format (AAA111)
    if re.match(r"^[A-Z]{3}\d{3}$", extracted_text):
        return extracted_text
    return f"Invalid: {extracted_text}"


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
    # OCR on processed image
    processed_img = processed_image(image_path)
    extracted_text = extract_text(processed_img)

    return processed_image(image_path), extracted_text
