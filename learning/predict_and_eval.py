# file: predict_and_eval.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt

class DronePlatformDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, row['image'])).convert('RGB')
        if self.transform:
            image = self.transform(image)

        target = row[1:4].astype(np.float32).values
        return image, torch.tensor(target)

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

def predict_single(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model = CNNMLP()
    model.load_state_dict(torch.load("cnn_mlp_model.pt"))
    model.eval()

    with torch.no_grad():
        pred = model(image)
    return pred.squeeze().numpy()

def evaluate(csv_path, img_dir):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = DronePlatformDataset(csv_path, img_dir, transform)
    loader = DataLoader(dataset, batch_size=32)

    model = CNNMLP()
    model.load_state_dict(torch.load("cnn_mlp_model.pt"))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for img, target in loader:
            pred = model(img)
            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    error = np.mean(np.abs(preds - targets), axis=0)
    print(f"Mean Absolute Error: x={error[0]:.3f}, y={error[1]:.3f}, z={error[2]:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(targets[:,0], targets[:,1], label='True', alpha=0.6)
    plt.scatter(preds[:,0], preds[:,1], label='Predicted', alpha=0.6)
    plt.xlabel("Rel X")
    plt.ylabel("Rel Y")
    plt.legend()
    plt.title("Relative Platform XY Prediction")
    plt.grid(True)
    plt.savefig("prediction_xy_plot.png")
    print("Saved prediction plot to prediction_xy_plot.png")

if __name__ == '__main__':
    print("Evaluating full dataset...")
    evaluate("dataset/dataset.csv", "dataset")

    test_image = "dataset/frame_000001.jpg"
    pred = predict_single(test_image)
    print("Single prediction (rel_x, rel_y, rel_z):", pred)
