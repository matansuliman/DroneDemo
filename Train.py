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


class DroneRelPosDataset(Dataset):
    def __init__(self, labels_csv: str, img_dir: str, transform=None):
        self.df = pd.read_csv(labels_csv)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        label = torch.tensor([row['rel_x'], row['rel_y'], row['rel_z']], dtype=torch.float32)
        return img, label


class RelPosRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(weights=None)  # Use pretrained=True if internet is available
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_model(dataset_dir='dataset', epochs=20, batch_size=32, lr=1e-4):
    csv_path = os.path.join(dataset_dir, 'labels.csv')
    dataset = DroneRelPosDataset(csv_path, dataset_dir)

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RelPosRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    log_path = os.path.join(dataset_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for img, label in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, label in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
                img, label = img.to(device), label.to(device)
                pred = model(img)
                loss = loss_fn(pred, label)
                val_loss += loss.item() * img.size(0)

        avg_train = train_loss / len(train_loader.dataset)
        avg_val = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch}: Train MSE={avg_train:.4f}, Val MSE={avg_val:.4f}")
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train, avg_val])

    torch.save(model.state_dict(), os.path.join(dataset_dir, 'relposnet_mobilenet.pt'))
    print("\n✅ Training complete. Model saved to relposnet_mobilenet.pt")


if __name__ == '__main__':
    train_model()