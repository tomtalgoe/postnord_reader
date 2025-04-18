import os
import cv2
import subprocess
import traceback
import json
import zipfile
from flask import Flask, request, send_file, jsonify
from podtnord_ocr import process_image, model, config, reload_model
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

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump({"model_folder": "train7"}, f)

def logline(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}(api) - {message}")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}(api) - {message}\n")


@app.route("/imageprocessing", methods=["POST"])
def imageprocessing():
    starttime = datetime.now()

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400
    logline(f"Received image: {image.filename}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    unique_filename = f"{timestamp}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    try:
        image.save(image_path)

        processed_img, extracted_text, roi_image, bbox = process_image(image_path)
        processed_image_filename = f"{timestamp}_box_{image.filename}"
        processed_image_path = os.path.join(PROCESSED_FOLDER, processed_image_filename)
        logline(f"Processed image saved as: {processed_image_path}")

        cv2.imwrite(
            processed_image_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )

        roi_image_filename = f"{timestamp}_roi_{image.filename}"
        roi_image_path = os.path.join(PROCESSED_FOLDER, roi_image_filename)
        cv2.imwrite(roi_image_path, roi_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        elapsed_ms = (datetime.now() - starttime).total_seconds() * 1000
        logline(f"Image processed successfully. Extracted text: {extracted_text}. Processing time: {elapsed_ms:.2f} ms")

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
    wrong_folder = os.path.join(DATA_FOLDER, "wrong", "original")
    files = [f for f in os.listdir(wrong_folder) if f.endswith(".jpg")]

    file_list_html = "<ul id='file-list'>"
    for file in files:
        timestamp, remaining = file.split("_")[0], file.split("_")[1].split(".")[0]
        formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d-%H%M%S-%f").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        link_text = f"{formatted_timestamp} - {remaining}"
        file_list_html += f"<li><a href='#' onclick=\"showImage('{timestamp}', '{remaining}', 'wrong')\" id=\"link-{timestamp}\">{link_text}</a></li>"
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
    correct_folder = os.path.join(DATA_FOLDER, "correct", "original")
    files = [f for f in os.listdir(correct_folder) if f.endswith(".jpg")]

    file_list_html = "<ul id='file-list'>"
    for file in files:
        timestamp, remaining = file.split("_")[0], file.split("_")[1].split(".")[0]
        formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d-%H%M%S-%f").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        link_text = f"{formatted_timestamp} - {remaining}"
        file_list_html += f"<li><a href='#' onclick=\"showImage('{timestamp}', '{remaining}', 'correct')\" id=\"link-{timestamp}\">{link_text}</a></li>"
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
    filename=upload_file.rsplit("_", 1)[-1].rsplit(".", 1)[0] if upload_file else None
    json_file=os.path.join(PROCESSED_FOLDER, timestamp + "_" + filename + ".json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            json_data = json.load(f)
    else:
        logline(f"JSON file not found for timestamp: {timestamp}, json_file: {json_file}")
        return jsonify({"error": "JSON file not found for timestamp: " + timestamp}), 404
    processed_roi_file = os.path.join(PROCESSED_FOLDER, timestamp + "_roi_" + filename + ".jpg")
    # get the file object of the filename
    file_object = os
    processed_file = os.path.join(PROCESSED_FOLDER, timestamp + "_box_" + filename + ".jpg")

    if not upload_file or not processed_file:
        logline(f"Files not found for timestamp: {timestamp}, upload_file: {upload_file}, processed_file: {processed_file}")
        return jsonify({"error": "Files not found for the given timestamp"}), 404

    label = "correct" if correct else "wrong"
    original_dest = os.path.join(DATA_FOLDER, label, "original")
    processed_dest = os.path.join(DATA_FOLDER, label, "processed")
    os.makedirs(original_dest, exist_ok=True)
    os.makedirs(processed_dest, exist_ok=True)

    src_upload = os.path.join(UPLOAD_FOLDER, upload_file)
    # src_processed = os.path.join(PROCESSED_FOLDER, processed_file)
    # Read the json file into a json object
    # delete the json file if exists
    if os.path.exists(json_file):
        os.remove(json_file)
    dst_upload = os.path.join(original_dest, upload_file)
    dst_processed = os.path.join(processed_dest, timestamp + "_box_" + filename + ".jpg")
    dst_roi_processed = os.path.join(processed_dest, timestamp + "_roi_" + filename + ".jpg")
    dst_json = os.path.join(processed_dest, processed_file.rsplit(".", 1)[0] + ".json")
    logline(f"Moved files: {src_upload} -> {dst_upload}, {processed_file} -> {dst_processed}, {processed_roi_file} -> {dst_roi_processed}, deleted {json_file}")
    # return jsonify({"message": f"Feedback saved to '{label}' set"}), 200

    os.rename(src_upload, dst_upload)
    os.rename(processed_file, dst_processed)
    os.rename(processed_roi_file, dst_roi_processed)

    # if correct:
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
            json_data["bbox"] = best_box
            json_data["class"] = 0  # Assuming class 0 for the correct label
            json_data["image"] = upload_file
            # annotation = {
            #     "bbox": best_box,
            #     "class": 0,
            #     "image": upload_file,
            #     "timestamp": timestamp,
            # }
            annotation_path = os.path.join(original_dest, upload_file.rsplit(".", 1)[0] + ".json")
            with open(annotation_path, "w") as f:
                json.dump(json_data, f)
            logline(f"Annotation JSON saved for {upload_file}")
    except Exception as e:
        logline(f"Failed to save json_data JSON: {e}")

    logline(f"Feedback recorded for {timestamp}: {'correct' if correct else 'wrong'}")
    return jsonify({"message": f"Feedback saved to '{label}' set"}), 200


@app.route("/log")
def view_log():
    try:
        with open(LOG_FILE, "r") as log_file:
            lines = log_file.readlines()
            last_lines = lines[-100:]

        # Get available model folders
        model_folders = [
            folder for folder in os.listdir(os.path.join("runs", "detect"))
            if os.path.isfile(os.path.join("runs", "detect", folder, "weights", "best.pt"))
        ]
        model_folders = sorted(model_folders, key=lambda x: int(x.replace("train", "0")), reverse=True)
        model_folders_html = "".join(
            f"<option value='{folder}' {'selected' if folder == config.get('model_folder') else ''}>{folder}</option>"
            for folder in model_folders
        )

        return generate_html("log", f"""
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 10px;">
            <a href="#" onclick="confirmRestart()">Update server</a>
            <div style="display: flex; align-items: center; gap: 10px;">
            <label for='model-folder'>Select Model:</label>
            <select name='model_folder' id='model-folder'>
            {model_folders_html}
            </select>
            <button onclick="updateModel()">Update Model</button>
            </div>
        </div>
        <h1>Log File</h1>
        <pre>{''.join(last_lines)}</pre>
        <script>
            function updateModel() {{
            const selectedModel = document.getElementById('model-folder').value;
            fetch('/config', {{
            method: 'POST',
            headers: {{
            'Content-Type': 'application/json',
            }},
            body: JSON.stringify({{ model_folder: selectedModel }})
            }})
            .then(response => response.json())
            .then(data => alert(data.message || 'Model updated successfully'))
            .catch(error => alert('Error: ' + error));
            }}
        </script>
        """)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/config", methods=["GET", "POST"])
def config_route():
    if request.method == "POST":
        data = request.json
        model_folder = data.get("model_folder")
        logline(f"Updating model folder to: {model_folder}")
        if model_folder:
            with open(config_path, 'r+') as f:
                config = json.load(f)
                config["model_folder"] = model_folder
                f.seek(0)
                json.dump(config, f)
                f.truncate()
            reload_model()  # Reload YOLO model with new configuration
            return jsonify({"message": "The new training model '{}' has been set.".format(model_folder)}), 200
        return jsonify({"error": "Invalid model folder."}), 400

    with open(config_path, 'r') as f:
        config = json.load(f)
    return jsonify(config)

# Custom error handler to suppress detailed error information
@app.errorhandler(Exception)
def handle_exception(e):
    logline(f"Error: {e}")
    return "", 204  # Return no content for any error

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
