import os
import cv2
import subprocess
import traceback
import json
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


@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), mimetype="image/jpeg")


@app.route("/update")
def update_server():
    try:
        repo_path = os.path.expanduser("~/labelref/postnord_reader")
        subprocess.run(["git", "-C", repo_path, "pull", "origin", "main"], check=True)
        return jsonify({"message": "Server updated!"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to update: {e}"}), 500


@app.route("/files")
def list_files():
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .container { display: flex; }
            .file-list { width: 50%; }
            .image-viewer { width: 50%; text-align: center; }
            img { max-width: 100%; height: auto; }
            ul { list-style-type: none; padding: 0; }
            li { margin-bottom: 10px; }
            a { text-decoration: none; color: blue; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Pending Files for Review</h1>
        <div class="container">
            <div class="file-list">
                <ul>
    """

    for meta_file in os.listdir(PROCESSED_FOLDER):
        if not meta_file.endswith(".json"):
            continue

        with open(os.path.join(PROCESSED_FOLDER, meta_file), "r") as f:
            meta = json.load(f)

        timestamp = meta["timestamp"]
        extracted_text = meta["extracted_text"]
        processing_time = meta.get("processing_time_ms", "N/A")
        image_name = next(
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

        html += f"<li><a href=\"#\" onclick=\"showImages('{image_name}', '{processed_file}', '{timestamp}', '{extracted_text}', '{processing_time}')\">Uploaded: {image_name or 'N/A'}<br>Processed: {processed_file} (⏱ {processing_time} ms)</a></li>"

    html += """
                </ul>
            </div>
            <div class="image-viewer" id="image-viewer">
                <h2>Select a file to view images</h2>
            </div>
        </div>
        <script>
            function showImages(uploadFile, processedFile, timestamp, extractedText, processingTime) {
                const viewer = document.getElementById('image-viewer');
                viewer.innerHTML = `
                    <h2>Images for ${timestamp}</h2>
                    <h3>Uploaded Image</h3>
                    <img src="/uploads/${uploadFile}" alt="Uploaded Image">
                    <h3>Processed Image</h3>
                    <img src="/processed/${processedFile}" alt="Processed Image">
                    <p>Extracted Text: ${extractedText}</p>
                    <p>Processing time: ${processingTime} ms</p>
                    <p>Timestamp: ${timestamp}</p>
                    <button onclick="sendFeedback('${timestamp}', true)">✅ Correct</button>
                    <button onclick="sendFeedback('${timestamp}', false)">❌ Wrong</button>
                    <div id="feedback-msg"></div>
                `;
            }

            function sendFeedback(timestamp, correct) {
                fetch('/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ timestamp, correct })
                })
                .then(response => response.json())
                .then(data => {
                    const msgBox = document.getElementById("feedback-msg");
                    if (data.message) {
                        msgBox.innerHTML = `<p style="color:green;">${data.message}</p>`;
                    } else {
                        msgBox.innerHTML = `<p style="color:red;">${data.error}</p>`;
                    }
                })
                .catch(error => {
                    document.getElementById("feedback-msg").innerHTML = `<p style="color:red;">Error: ${error}</p>`;
                });
            }
        </script>
    </body>
    </html>
    """
    return html


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
            return f"<html><body><h1>Log File</h1><pre>{''.join(last_lines)}</pre></body></html>"
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
