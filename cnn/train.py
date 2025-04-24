import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd



# --- Config ---
CSV_FILE = "/mnt/data/sanjay/captcha_dataset/dataset/captcha_data.csv"
ROOT_DIR = "/mnt/data/sanjay/captcha_dataset/dataset"
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptchaDataset(Dataset):
    def __init__(self, csv_file, split="train", root_dir="", transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        if split == "train":
            self.data = self.data[self.data["image_path"].str.startswith("train-images")]
        elif split == "val":
            self.data = self.data[self.data["image_path"].str.startswith("validation-images")]
        elif split == "test":
            self.data = self.data[self.data["image_path"].str.startswith("test-images")]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row['image_path'])
        label_str = str(row['solution']).zfill(6)

        try:
            image = Image.open(image_path).convert("L")
            if self.transform:
                image = self.transform(image)
            label = torch.tensor([int(c) for c in label_str], dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"[ERROR] Skipping {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))


def get_loaders(csv_file, root_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.RandomRotation(5),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CaptchaDataset(csv_file, split="train", root_dir=root_dir + "/train-images", transform=train_transform)
    val_dataset = CaptchaDataset(csv_file, split="val", root_dir=root_dir + "/validation-images", transform=val_transform)

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

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

def compute_accuracy(preds, labels):
    pred_digits = preds.argmax(dim=2)
    correct = (pred_digits == labels).float()
    per_digit_acc = correct.mean().item()
    full_seq_acc = (correct.sum(dim=1) == 6).float().mean().item()
    return per_digit_acc, full_seq_acc


def save_checkpoint(model, epoch, accuracy, path):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'config': {
            'epochs': epoch + 1,
            'accuracy': accuracy,
            'model': str(model),
        }
    }, path)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler):
    best_acc = 0.0
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}] Training", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs.permute(0, 2, 1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

        # --- Validation ---
        model.eval()
        total_per_digit_acc = 0
        total_full_seq_acc = 0
        val_batches = 0
        val_bar = tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)

                per_digit_acc, full_seq_acc = compute_accuracy(outputs, labels)
                total_per_digit_acc += per_digit_acc
                total_full_seq_acc += full_seq_acc
                val_batches += 1

                val_bar.set_postfix(per_digit_acc=per_digit_acc, full_seq_acc=full_seq_acc)

        avg_per_digit_acc = total_per_digit_acc / val_batches
        avg_full_seq_acc = total_full_seq_acc / val_batches

        tqdm.write(f"[Epoch {epoch+1}] Metrics | Loss: {avg_loss:.4f} | Per-digit acc: {avg_per_digit_acc:.4f} | Full-seq acc: {avg_full_seq_acc:.4f}")

        # --- Save Best Model ---
        if avg_full_seq_acc > best_acc:
            best_acc = avg_full_seq_acc
            save_checkpoint(model, epoch, best_acc, "captcha_cnn.pth")
            patience = 0
        else:
            patience += 1

        scheduler.step()

        if patience >= 5:
            print(f" Early stopping at epoch {epoch+1}")
            break



if __name__ == "__main__":
    train_loader, val_loader = get_loaders(CSV_FILE, ROOT_DIR, BATCH_SIZE)
    model = CaptchaCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler)
