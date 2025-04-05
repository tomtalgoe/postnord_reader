import os
import cv2
import subprocess
import traceback
import json
import zipfile
from flask import Flask, request, send_file, jsonify
from podtnord_ocr import process_image, model
from datetime import datetime

app = Flask(__name__)

# Folder structure setup
DATA_FOLDER = "data"
PENDING_FOLDER = os.path.join(DATA_FOLDER, "pending")
UPLOAD_FOLDER = os.path.join(PENDING_FOLDER, "original")
PROCESSED_FOLDER = os.path.join(PENDING_FOLDER, "processed")
LOG_FILE = "../server.log"

os.makedirs(PENDING_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def logline(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}(api) - {message}")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}(api) - {message}\n")


@app.route("/imageprocessing", methods=["POST"])
def imageprocessing():
    starttime = datetime.now()
    logline("Image Processing API, files: {}".format(request.files))

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400
    logline(f"Received image: {image.filename}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    logline(f"Timestamp: {timestamp}")
    unique_filename = f"{timestamp}_{image.filename}"
    logline(f"Unique filename: {unique_filename}")
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    logline(f"Saving image to: {image_path}")
    try:
        image.save(image_path)
        logline(f"Image saved successfully: {image_path}")

        processed_img, extracted_text, roi_image, bbox = process_image(image_path)
        logline(f"Processed image: {processed_img.shape}")
        processed_image_filename = f"{timestamp}_{extracted_text[:10]}_{image.filename}"
        logline(f"Processed image filename: {processed_image_filename}")
        processed_image_path = os.path.join(PROCESSED_FOLDER, processed_image_filename)
        logline(f"Processed image saved as: {processed_image_path}")

        cv2.imwrite(
            processed_image_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )

        roi_image_filename = f"{timestamp}_roi_{image.filename}"
        roi_image_path = os.path.join(PROCESSED_FOLDER, roi_image_filename)
        cv2.imwrite(roi_image_path, roi_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        logline(f"ROI image saved as: {roi_image_path}")

        elapsed_ms = (datetime.now() - starttime).total_seconds() * 1000
        logline(
            f"Image processed successfully. Extracted text: {extracted_text}. Processing time: {elapsed_ms:.2f} ms"
        )

        # Save metadata JSON
        metadata = {
            "timestamp": timestamp,
            "filename": image.filename,
            "extracted_text": extracted_text,
            "processing_time_ms": round(elapsed_ms, 2),
        }
        metadata_filename = f"{timestamp}_{image.filename.rsplit('.', 1)[0]}.json"
        metadata_path = os.path.join(PROCESSED_FOLDER, metadata_filename)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        bbox = [int(x) for x in bbox] if bbox else None

        return jsonify(
            {
                "timestamp": timestamp,
                "text": extracted_text,
                "processed_image_url": f"/processed/{os.path.basename(processed_image_path)}",
                "uploaded_image_url": f"/uploads/{os.path.basename(image_path)}",
                "roi_image_url": f"/processed/{roi_image_filename}",
                "bbox": bbox,
                "processing_time_ms": round(elapsed_ms, 2),
            }
        )

    except Exception as e:
        logline(f"Error processing image: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<filename>")
def get_uploaded_image(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route("/wrong/<filename>")
def get_wrong_image(filename):
    if os.path.exists(os.path.join(DATA_FOLDER, "wrong", "processed", filename)):
        return send_file(os.path.join(DATA_FOLDER, "wrong", "processed", filename), mimetype="image/jpeg")
    return send_file(os.path.join(DATA_FOLDER, "wrong", "original", filename), mimetype="image/jpeg")

@app.route("/wrong/download_all")
def download_wrong_images():
    zip_filename = "wrong_images.zip"
    zip_path = os.path.join(DATA_FOLDER, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, "wrong", "original")):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.join(DATA_FOLDER, "wrong"))
                zipf.write(file_path, arcname)
    logline(f"Created wrong ZIP file: {zip_path} with length {os.path.getsize(zip_path)} bytes")
    return send_file(zip_path, as_attachment=True)

@app.route("/wrong/remove_all")
def remove_wrong_images():
    try:
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, "wrong")):
            for file in files:
                os.remove(os.path.join(root, file))
        logline("All wrongly classified images removed successfully.")
        return jsonify({"message": "All wrongly classified images removed successfully."}), 200
    except Exception as e:
        logline(f"Error removing wrongly classified images: {e}")
        return jsonify({"error": str(e)}), 500

from flask import send_file
from common import generate_navbar, generate_html

@app.route("/wrong")
def list_wrong_images():
    wrong_folder = os.path.join(DATA_FOLDER, "wrong", "processed")
    files = [f for f in os.listdir(wrong_folder) if f.endswith(".jpg")]

    file_list_html = "<ul id='file-list'>"
    for file in files:
        timestamp, extracted, remaining = file.split("_")[0], file.split("_")[1], file.split("_")[2].split(".")[0]
        formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d-%H%M%S-%f").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        link_text = f"{formatted_timestamp} - {extracted} - {remaining}"
        file_list_html += f"<li><a href='#' onclick=\"showImage('{timestamp}', '{extracted}', '{remaining}', 'wrong')\" id=\"link-{timestamp}\">{link_text}</a></li>"
    file_list_html += "</ul>"

    content = f"""
    <h1>Wrongly Classified Images</h1>
    <div class='container'>
        <div class='file-list'>
            {file_list_html}
            <button onclick="window.location.href='/wrong/download_all'">Download All</button>
            <button onclick="window.location.href='/wrong/remove_all'">Remove All</button>
        </div>
        <div class='image-viewer' id='image-viewer'>
            <h2>Select an image to view details</h2>
        </div>
    </div>
    """

    return generate_html("wrong", content)

@app.route("/correct/<filename>")
def get_correct_image(filename):
    if os.path.exists(os.path.join(DATA_FOLDER, "correct", "processed", filename)):
        return send_file(os.path.join(DATA_FOLDER, "correct", "processed", filename), mimetype="image/jpeg")
    return send_file(os.path.join(DATA_FOLDER, "correct", "original", filename), mimetype="image/jpeg")

@app.route("/correct/download_all")
def download_correct_images():
    zip_filename = "correct_images.zip"
    zip_path = os.path.join(DATA_FOLDER, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, "wrong", "original")):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.join(DATA_FOLDER, "wrong"))
                zipf.write(file_path, arcname)
    logline(f"Created correct ZIP file: {zip_path} with length {os.path.getsize(zip_path)} bytes")
    return send_file(zip_path, as_attachment=True)

@app.route("/correct/remove_all")
def remove_correct_images():
    try:
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, "correct")):
            for file in files:
                os.remove(os.path.join(root, file))
        logline("All correctly classified images removed successfully.")
        return jsonify({"message": "All correctly classified images removed successfully."}), 200
    except Exception as e:
        logline(f"Error removing correctly classified images: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/correct")
def list_correct_images():
    correct_folder = os.path.join(DATA_FOLDER, "correct", "processed")
    files = [f for f in os.listdir(correct_folder) if f.endswith(".jpg")]

    file_list_html = "<ul id='file-list'>"
    for file in files:
        timestamp, extracted, remaining = file.split("_")[0], file.split("_")[1], file.split("_")[2].split(".")[0]
        formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d-%H%M%S-%f").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        link_text = f"{formatted_timestamp} - {extracted} - {remaining}"
        file_list_html += f"<li><a href='#' onclick=\"showImage('{timestamp}', '{extracted}', '{remaining}', 'correct')\" id=\"link-{timestamp}\">{link_text}</a></li>"
    file_list_html += "</ul>"

    content = f"""
    <h1>Correctly Classified Images</h1>
    <div class='container'>
        <div class='file-list'>
            {file_list_html}
            <button onclick="window.location.href='/correct/download_all'">Download All</button>
            <button onclick="window.location.href='/correct/remove_all'">Remove All</button>
        </div>
        <div class='image-viewer' id='image-viewer'>
            <h2>Select an image to view details</h2>
        </div>
    </div>
    """

    return generate_html("correct", content)

@app.route("/correct/original/<filename>")
def get_correct_file(filename):
    return send_file(os.path.join(DATA_FOLDER, "correct", "original", filename))

@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), mimetype="image/jpeg")

# Page showing all the wrongly classified images. The user can select one and see the image and the extracted text.
# It should also be possible to download the original image and one for downloading all the wrongly classified images as a zip file.
@app.route("/files")
def list_files():
    navbar = generate_navbar("files")

    file_list_html = "<ul id='file-list'>"
    for meta_file in os.listdir(PROCESSED_FOLDER):
        if not meta_file.endswith(".json"):
            continue

        with open(os.path.join(PROCESSED_FOLDER, meta_file), "r") as f:
            meta = json.load(f)

        timestamp = meta["timestamp"]
        filename= meta['filename']
        extracted_text = meta["extracted_text"]
        processing_time = meta.get("processing_time_ms", "N/A")
        formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d-%H%M%S-%f").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        link_text = f"{formatted_timestamp} - {extracted_text} (⏱ {processing_time} ms)"

        file_list_html += f"<li><a href='#' onclick=\"showFile('{timestamp}', '{extracted_text}', '{filename}')\" id=\"link-{timestamp}\">{link_text}</a></li>"
    file_list_html += "</ul>"

    content = f"""
    <h1>Pending Files for Review</h1>
    <div class='container'>
        <div class='file-list'>
            {file_list_html}
        </div>
        <div class='image-viewer' id='image-viewer'>
            <h2>Select a file to view details</h2>
        </div>
    </div>
    <script>
        function showFile(timestamp, extractedText, filename) {{
            const viewer = document.getElementById('image-viewer');
            const links = document.querySelectorAll('#file-list a');
            links.forEach(link => link.classList.remove('selected'));
            document.getElementById(`link-${{timestamp}}`).classList.add('selected');

            const uploadFile = `${{timestamp}}_${{filename}}`;
            const processedFile = `${{timestamp}}_${{extractedText}}_${{filename}}`;

            viewer.innerHTML = `
                <h2>File Details</h2>
                <h3>Uploaded Image</h3>
                <img src="/uploads/${{uploadFile}}" alt="Uploaded Image">
                <h3>Processed Image</h3>
                <img src="/processed/${{processedFile}}" alt="Processed Image">
                <p>Extracted Text: ${{extractedText}}</p>
                <p>Filename: ${{filename}}</p>
            `;
        }}
    </script>
    """

    return generate_html(navbar, content)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    timestamp = data.get("timestamp")
    correct = data.get("correct")

    if not timestamp or correct is None:
        return jsonify({"error": "Missing 'timestamp' or 'correct' in request"}), 400

    upload_file = next(
        (f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(timestamp)), None
    )
    processed_file = next(
        (
            f
            for f in os.listdir(PROCESSED_FOLDER)
            if f.startswith(timestamp) and not f.endswith(".json")
        ),
        None,
    )

    if not upload_file or not processed_file:
        return jsonify({"error": "Files not found for the given timestamp"}), 404

    label = "correct" if correct else "wrong"
    original_dest = os.path.join(DATA_FOLDER, label, "original")
    processed_dest = os.path.join(DATA_FOLDER, label, "processed")
    os.makedirs(original_dest, exist_ok=True)
    os.makedirs(processed_dest, exist_ok=True)

    src_upload = os.path.join(UPLOAD_FOLDER, upload_file)
    src_processed = os.path.join(PROCESSED_FOLDER, processed_file)
    dst_upload = os.path.join(original_dest, upload_file)
    dst_processed = os.path.join(processed_dest, processed_file)

    os.rename(src_upload, dst_upload)
    os.rename(src_processed, dst_processed)

    if correct:
        try:
            image = cv2.imread(dst_upload)
            results = model(image)[0]

            if results.boxes is not None and results.boxes.xyxy.shape[0] > 0:
                best_box = (
                    results.boxes.xyxy[results.boxes.conf.argmax()]
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )
                annotation = {
                    "bbox": best_box,
                    "class": 0,
                    "image": upload_file,
                    "timestamp": timestamp,
                }
                annotation_path = os.path.join(
                    original_dest, upload_file.rsplit(".", 1)[0] + ".json"
                )
                with open(annotation_path, "w") as f:
                    json.dump(annotation, f)
                logline(f"Annotation JSON saved for {upload_file}")
        except Exception as e:
            logline(f"Failed to save annotation JSON: {e}")

    logline(f"Feedback recorded for {timestamp}: {'correct' if correct else 'wrong'}")
    return jsonify({"message": f"Feedback saved to '{label}' set"}), 200


@app.route("/log")
def view_log():
    try:
        with open(LOG_FILE, "r") as log_file:
            lines = log_file.readlines()
            last_lines = lines[-100:]
            return generate_html("log", f"""<h1>Log File</h1>
                <pre>{''.join(last_lines)}</pre>
                <a href="#" onclick="confirmRestart()">Update server</a>
                <script>
                    function confirmRestart() {{
                        if (confirm("Are you sure you want the server to fetch latest checkedin code and then restart the server?")) {{
                            window.location.href = '/update';
                        }}
                    }}
                </script>""")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
