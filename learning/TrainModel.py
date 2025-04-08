# file: train_model.py
import os
import csv
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import mobilenet_v2
from tqdm import tqdm

class DronePlatformDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(row[1:4].values.astype(np.float32))
        return image, target

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, times=2):
        self.base = base_dataset
        self.times = times

    def __len__(self):
        return len(self.base) * self.times

    def __getitem__(self, idx):
        base_idx = idx % len(self.base)
        image, target = self.base[base_idx]
        return image, target

class CNNMLP(nn.Module):
    def __init__(self, cnn_out_dim=1280):
        super().__init__()
        base = mobilenet_v2(pretrained=True).features
        self.cnn = nn.Sequential(base, nn.AdaptiveAvgPool2d((1, 1)))
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, image):
        x = self.cnn(image).view(image.size(0), -1)
        return self.mlp(x)

def main():
    dataset_dir = "dataset"
    csv_path = os.path.join(dataset_dir, "dataset.csv")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    base_dataset = DronePlatformDataset(csv_path, dataset_dir, transform)
    augmented_dataset = base_dataset #AugmentedDataset(base_dataset, times=2)

    train_idx, val_idx = train_test_split(list(range(len(augmented_dataset))), test_size=0.2, random_state=42)
    train_set = Subset(augmented_dataset, train_idx)
    val_set = Subset(augmented_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    log_path = "training_log.csv"
    with open(log_path, 'w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(10):
        model.train()
        total_train_loss = 0
        for img, target in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img, target in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                img, target = img.to(device), target.to(device)
                pred = model(img)
                loss = criterion(pred, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        with open(log_path, 'a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([epoch+1, avg_train_loss, avg_val_loss])

    torch.save(model.state_dict(), "cnn_mlp_model.pt")
    print("Model saved as cnn_mlp_model.pt")
    print(f"Training log saved to {log_path}")

if __name__ == '__main__':
    main()
