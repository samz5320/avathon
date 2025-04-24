import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# ------------------- Config -------------------
IMG_SIZE = (100, 32) 
CHARACTERS = '0123456789'
CHAR_MAP = {c: i for i, c in enumerate(CHARACTERS)}
INV_CHAR_MAP = {v: k for k, v in CHAR_MAP.items()}
NUM_CLASSES = len(CHARACTERS) + 1  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = "/mnt/data/sanjay/captcha_dataset/dataset/validation-images/validation-images/image_validation_1.png"
MODEL_PATH = "/Users/apple/Downloads/avathon/cnn/avathon/crnn/best_crnn_model.pth"

# ------------------- Model -------------------
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(128 * 8, 128, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x) 
        b, c, h, w = x.size()
        x = x.permute(3, 0, 2, 1).contiguous().view(w, b, -1)  
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# ------------------- Decode -------------------
def decode_predictions(preds):
    preds = preds.permute(1, 0, 2)  
    preds = torch.argmax(preds, dim=2).cpu().numpy()
    results = []
    for pred in preds:
        seq = []
        prev = -1
        for p in pred:
            if p != prev and p != NUM_CLASSES - 1:
                seq.append(INV_CHAR_MAP.get(p, '?'))
            prev = p
        results.append(''.join(seq))
    return results

# ------------------- Predict -------------------
def predict_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)  

    with torch.no_grad():
        preds = model(img).log_softmax(2)
        decoded = decode_predictions(preds)[0]
        print(f"[PREDICT] {os.path.basename(image_path)} → {decoded}")
        return decoded

if __name__ == "__main__":
    model = CRNN().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predict_image(model, IMAGE_PATH)