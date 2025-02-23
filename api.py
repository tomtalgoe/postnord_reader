import os
import cv2
from flask import Flask, request, send_file, jsonify
from podtnord_ocr import process_image
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)  # Ensure processed folder exists

def logline(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

@app.route("/imageprosessing", methods=["POST"])
def imageprosessing():
    starttime = datetime.now()
    logline("Image Processing API, files: {}".format(request.files))

    # Check if image is in request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]

    # Check if file is empty
    if image.filename == "":
        logline("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Save the image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        # Process the image and extract text
        processed_img, extracted_text = process_image(
            image_path
        )  # ✅ Now gets OCR text too

        processed_image_path = os.path.join(
            PROCESSED_FOLDER,
            "processed_" + os.path.splitext(image.filename)[0] + ".jpeg",
        )

        # Save the processed image as JPEG
        # Save with time stamp as filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_image_path = os.path.join(
            PROCESSED_FOLDER,
            f"processed_{timestamp}_{os.path.splitext(image.filename)[0]}.jpeg",
        )

        cv2.imwrite(
            processed_image_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )

        # Return JSON response with text and image URL
        
        logline("Image processed successfully, extracted text: {}, total time: {:.2f} ms".format(extracted_text, (datetime.now() - starttime).total_seconds() * 1000))
        response_data = {
            "text": extracted_text,
            "processed_image_url": f"/processed/{os.path.basename(processed_image_path)}",
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route to serve uploaded images
@app.route("/uploads/<filename>")
def get_uploaded_image(filename):
    if filename == "*":
        html = "<html><body><h1>All upload files</h1>"
        for file in os.listdir(UPLOAD_FOLDER):
            html += f'<a href="{file}">{file}</a><br>'
        html += "</body></html>"
        return html
    """Allows downloading/viewing uploaded images."""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


# Route to serve processed images
@app.route("/processed/<filename>")
def get_processed_image(filename):
    if filename == "*":
        html = "<html><body><h1>All processed files</h1>"
        for file in os.listdir(PROCESSED_FOLDER):
            html += f'<a href="{file}">{file}</a><br>'
        html += "</body></html>"
        return html
    """Allows downloading/viewing processed images."""
    return send_file(os.path.join(PROCESSED_FOLDER, filename), mimetype="image/jpeg")

@app.route('/update', methods=['POST'])
def update_server():
    try:
        # Navigate to project directory
        repo_path = os.path.expanduser("~/labelref/postnord_reader")
        subprocess.run(["git", "-C", repo_path, "pull", "origin", "main"], check=True)

        return jsonify({"message": "Server updated!"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to update: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
