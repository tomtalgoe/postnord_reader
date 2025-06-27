import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, url_for
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 1️⃣ Character set for your model
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
SEQ_LENGTH = 6
VOCAB_SIZE = len(CHARS)

# 2️⃣ Your model architecture
class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 16 * 16, 512)
        self.relu = nn.ReLU()
        self.heads = nn.ModuleList([nn.Linear(512, VOCAB_SIZE) for _ in range(SEQ_LENGTH)])

    def forward(self, x):
        features = self.cnn(x)
        features = self.flatten(features)
        features = self.relu(self.fc(features))
        outputs = [head(features) for head in self.heads]
        return outputs

# 3️⃣ Inference function for your model
def predict_custom(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        predictions = [torch.argmax(o, dim=1).item() for o in outputs]
        predicted_label = ''.join([IDX_TO_CHAR[idx] for idx in predictions])
    return predicted_label

# 4️⃣ Inference function for TrOCR
def predict_trocr(trocr_processor, trocr_model, image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    predicted_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text

# 5️⃣ Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your model
custom_model = OCRModel()
custom_model.load_state_dict(torch.load("ocr_model.pth", map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model.to(device)

# Load TrOCR model
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Web route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_custom = None
    prediction_trocr = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        prediction_custom = predict_custom(custom_model, filepath, device)
        prediction_trocr = predict_trocr(trocr_processor, trocr_model, filepath)
        image_url = url_for('static', filename=f"uploads/{filename}")
        return render_template("demo.html", 
                               prediction_custom=prediction_custom, 
                               prediction_trocr=prediction_trocr,
                               image_url=image_url)

    return render_template("demo.html")

if __name__ == "__main__":
    app.run(debug=True)
