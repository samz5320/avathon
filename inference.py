import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# --- Model Definition (same as training) ---
class CaptchaCNN(nn.Module):
    def __init__(self, num_digits=6, num_classes=10):
        super(CaptchaCNN, self).__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_digits * num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x.view(-1, self.num_digits, self.num_classes)

# --- Load Model ---
def load_model(path, device):
    model = CaptchaCNN().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# --- Preprocess Image ---
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert("L")
    return transform(img).unsqueeze(0)  # [1, 1, 50, 200]

# --- Inference Function ---
def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred_digits = output.argmax(dim=2).squeeze(0).cpu().tolist()
        return ''.join(str(d) for d in pred_digits)

# --- Entry Point ---
if __name__ == "__main__":
    import argparse

    model="model.pth"
    img="dataset/validation-images/validation-images/image_validation_1.png"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, DEVICE)
    image_tensor = preprocess_image(img)
    prediction = predict(model, image_tensor, DEVICE)

    print(f"Predicted CAPTCHA: {prediction}")