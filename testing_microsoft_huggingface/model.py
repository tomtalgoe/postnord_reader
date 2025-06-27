import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# 1️⃣ Character set and mappings
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(CHARS)
SEQ_LENGTH = 6  # Always 6 characters

# 2️⃣ Dataset class
class OCRDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file, header=None, names=["filename", "label"])
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, row["filename"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = [CHAR_TO_IDX[c] for c in row["label"]]
        return image, torch.tensor(label, dtype=torch.long)

# 3️⃣ Model architecture
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

# 4️⃣ Loss function

def compute_loss(outputs, targets):
    loss = 0
    for i in range(SEQ_LENGTH):
        loss += nn.CrossEntropyLoss()(outputs[i], targets[:, i])
    return loss

# 5️⃣ Training function

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 6️⃣ Evaluation function

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.stack([torch.argmax(o, dim=1) for o in outputs], dim=1)
            total_correct += (predictions == labels).all(dim=1).sum().item()
            total_samples += images.size(0)
    return total_correct / total_samples

# 7️⃣ Full pipeline
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = OCRDataset(csv_file="testing_microsoft_huggingface\labels.csv", image_folder="testing_microsoft_huggingface\static\images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        train_loss = train(model, dataloader, optimizer, device)
        accuracy = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Accuracy = {accuracy*100:.2f}%")

    torch.save(model.state_dict(), "ocr_model.pth")
    print("✅ Training complete. Model saved.")
