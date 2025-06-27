from flask import Flask, request, render_template, redirect, send_from_directory
import os
import csv

app = Flask(__name__)

# Config
IMAGE_FOLDER = "testing_microsoft_huggingface\static\images"
LABELS_FILE = "labels.csv"

# Utility: load labeled files
def load_labeled():
    if not os.path.exists(LABELS_FILE):
        return set()
    with open(LABELS_FILE, newline='') as f:
        return set(row[0] for row in csv.reader(f))

# Main route
@app.route("/", methods=["GET", "POST"])
def label():
    labeled = load_labeled()
    all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    unlabeled_images = [img for img in all_images if img not in labeled]

    if request.method == "POST":
        filename = request.form["filename"]
        label = request.form["label"].strip().upper()

        if len(label) != 6:
            return f"Error: Label must be exactly 6 characters. You entered: '{label}'"

        with open(LABELS_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, label])

        return redirect("/")

    if unlabeled_images:
        current_image = unlabeled_images[0]
        return render_template("label.html", image=current_image)
    else:
        return "✅ All images labeled!"

# Serve images directly from images/ folder
@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
