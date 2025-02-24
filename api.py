import os
import cv2
import subprocess
from flask import Flask, request, send_file, jsonify
from podtnord_ocr import process_image
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
LOG_FILE = "../server.log"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)  # Ensure processed folder exists

def logline(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}(api) - {message}")

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

    # Save the image temporarily with a unique filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    unique_filename = f"{timestamp}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    image.save(image_path)

    try:
        # Process the image and extract text
        processed_img, extracted_text = process_image(image_path)

        processed_image_filename = f"{timestamp}_{extracted_text[:10]}_{image.filename}"
        processed_image_path = os.path.join(PROCESSED_FOLDER, processed_image_filename)

        # Save the processed image as JPEG
        cv2.imwrite(processed_image_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Return JSON response with text and image URL
        logline("Image processed successfully, extracted text: {}, total time: {:.2f} ms".format(extracted_text, (datetime.now() - starttime).total_seconds() * 1000))
        response_data = {
            "text": extracted_text,
            "processed_image_url": f"/processed/{os.path.basename(processed_image_path)}",
            "uploaded_image_url": f"/uploads/{os.path.basename(image_path)}",
        }

        return jsonify(response_data)

    except Exception as e:
        logline(f"Error processing image: {e}")
        # print stacktrace
        traceback.print_exc()
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

@app.route('/update')
def update_server():
    try:
        # Navigate to project directory
        repo_path = os.path.expanduser("~/labelref/postnord_reader")
        subprocess.run(["git", "-C", repo_path, "pull", "origin", "main"], check=True)
        logline("Server updated successfully")
        return jsonify({"message": "Server updated!"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to update: {e}"}), 500

@app.route('/files')
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
        <h1>Uploaded and Processed Files</h1>
        <div class="container">
            <div class="file-list">
                <ul>
    """
    for upload_file in os.listdir(UPLOAD_FOLDER):
        timestamp, original_filename = upload_file.split('_', 1)
        processed_file = next((f for f in os.listdir(PROCESSED_FOLDER) if f.startswith(timestamp)), None)
        if processed_file:
            extracted_text = processed_file.split('_')[1]
            html += f'<li><a href="#" onclick="showImages(\'{upload_file}\', \'{processed_file}\', \'{timestamp}\', \'{extracted_text}\')">Uploaded: {upload_file} (Timestamp: {timestamp})<br>Processed: {processed_file} (Extracted Text: {extracted_text})</a></li>'
    html += """
                </ul>
            </div>
            <div class="image-viewer" id="image-viewer">
                <h2>Select a file to view images</h2>
            </div>
        </div>
        <script>
            function showImages(uploadFile, processedFile, timestamp, extractedText) {
                const viewer = document.getElementById('image-viewer');
                viewer.innerHTML = `
                    <h2>Images for ${timestamp}</h2>
                    <h3>Uploaded Image</h3>
                    <img src="/uploads/${uploadFile}" alt="Uploaded Image">
                    <h3>Processed Image</h3>
                    <img src="/processed/${processedFile}" alt="Processed Image">
                    <p>Extracted Text: ${extractedText}</p>
                    <p>Timestamp: ${timestamp}</p>
                `;
            }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/view/<timestamp>')
def view_images(timestamp):
    upload_file = next((f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(timestamp)), None)
    processed_file = next((f for f in os.listdir(PROCESSED_FOLDER) if f.startswith(f"{timestamp}")), None)
    if upload_file and processed_file:
        extracted_text = processed_file.split('_')[1]
        html = f"<html><body><h1>Images for {timestamp}</h1>"
        html += f'<h2>Uploaded Image</h2><img src="/uploads/{upload_file}" alt="Uploaded Image"><br>'
        html += f'<h2>Processed Image</h2><img src="/processed/{processed_file}" alt="Processed Image"><br>'
        html += f'<p>Extracted Text: {extracted_text}</p>'
        html += f'<p>Timestamp: {timestamp}</p>'
        html += "</body></html>"
        return html
    return "Files not found", 404

@app.route('/log')
def view_log():
    try:
        with open(LOG_FILE, "r") as log_file:
            lines = log_file.readlines()
            last_lines = lines[-100:]
            html = "<html><body><h1>Log File</h1><pre>{}</pre></body></html>".format("".join(last_lines))
            return html
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
