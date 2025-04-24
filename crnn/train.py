# -------------------imports-------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from torch.nn import CTCLoss
import os

# -------------------config-------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = (100, 32) 
BATCH_SIZE = 32
EPOCHS = 50
CHARACTERS = '0123456789'
CHAR_MAP = {c: i for i, c in enumerate(CHARACTERS)}
INV_CHAR_MAP = {v: k for k, v in CHAR_MAP.items()}
NUM_CLASSES = len(CHARACTERS) + 1  

# -------------------utils-------------------
def preprocess_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

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

# -------------------dataset-------------------
class CaptchaDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode="train"):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mode = mode

        if mode == "train":
            self.data = self.data[self.data["image_path"].str.contains("train-images")]
        elif mode == "val":
            self.data = self.data[self.data["image_path"].str.contains("validation-images")]
        else:
            raise ValueError("mode must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        label = [CHAR_MAP[c] for c in str(row['solution'])]
        img = preprocess_image(img_path)
        return torch.tensor(img), torch.tensor(label), str(row['solution'])


# -------------------model-------------------
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * 8, 128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)  
        b, c, h, w = x.size()
        x = x.permute(3, 0, 2, 1).contiguous().view(w, b, -1)  
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x  

# -------------------collate_fn-------------------
def collate_fn(batch):
    imgs, labels, texts = zip(*batch)
    imgs = torch.stack(imgs)
    flat_labels = torch.cat(labels)
    input_lengths = torch.full((len(batch),), imgs.shape[-1] // 4, dtype=torch.long)
    target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    return imgs, flat_labels, input_lengths, target_lengths, texts


# -------------------train-------------------
def train(model, train_dl, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, targets, input_lengths, target_lengths, _ in train_dl:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        preds = model(imgs).log_softmax(2)
        loss = criterion(preds, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dl)

# -------------------validate-------------------
def validate(model, val_dl):
    model.eval()
    seq_correct, total = 0, 0
    digit_correct = 0
    total_digits = 0

    with torch.no_grad():
        for imgs, _, _, _, true_texts in val_dl:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).log_softmax(2)
            decoded = decode_predictions(preds)

            for pred, true in zip(decoded, true_texts):
                total += 1
                if pred == true:
                    seq_correct += 1

                min_len = min(len(pred), len(true))
                digit_correct += sum(p == t for p, t in zip(pred, true[:min_len]))
                total_digits += 6  

    print(f"[Validation] Sequence Accuracy: {seq_correct/total:.4f}, Digit Accuracy: {digit_correct/total_digits:.4f}")

# -------------------run-------------------
if __name__ == "__main__":
    model = CRNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)

    train_ds = CaptchaDataset("/mnt/data/sanjay/captcha_dataset/dataset/captcha_data.csv", "/mnt/data/sanjay/captcha_dataset/dataset/train-images/", mode="train")
    val_ds = CaptchaDataset("/mnt/data/sanjay/captcha_dataset/dataset/captcha_data.csv", "/mnt/data/sanjay/captcha_dataset/dataset/validation-images/", mode="val")

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples:   {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    for epoch in range(EPOCHS):
        loss = train(model, train_dl, optimizer, criterion)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
        validate(model, val_dl)

    torch.save(model.state_dict(), "crnn_ctc.pth")

