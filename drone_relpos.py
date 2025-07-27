
from __future__ import annotations

import argparse
import csv
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# --- Simulation imports (from your project) ---------------------------------
from orchestrator import Trainer  # uses MuJoCo & your own classes
import mujoco                     # needed to terminate cleanly

# Optional, only imported when training
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, random_split
    from torchvision import models
except ModuleNotFoundError:
    torch = None  # handled later

# -----------------------------------------------------------------------------
FPS = 5                     # camera FPS when capturing
IMG_WIDTH, IMG_HEIGHT = 320, 240  # resolution must match CameraStreamer

# -----------------------------------------------------------------------------
# 1. DATA‑CAPTURE STREAMER (stand‑alone, no Qt dependency) ---------------------
# -----------------------------------------------------------------------------

from streamer import CameraStreamer  # we override _run to avoid PyQt signals


class CaptureStreamer(CameraStreamer):
    """A headless camera streamer that saves frames + labels on the fly."""

    def __init__(
        self,
        orchestrator: Trainer,
        attached_body,
        save_dir: str | Path,
        n_samples: int,
        update_rate: int = FPS,
    ):
        super().__init__(orchestrator, attached_body, update_rate)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.save_dir / "labels.csv"
        self.csv_lock = threading.Lock()
        self.n_samples = n_samples
        self.counter = 0

        # Prepare CSV
        if not self.labels_path.exists():
            with open(self.labels_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "rel_x", "rel_y", "rel_z"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_sample(self, frame: np.ndarray):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_name = f"{timestamp}.png"
        img_path = self.save_dir / img_name

        # Save the RGB frame (PIL handles uint8 ndarray directly)
        Image.fromarray(frame).save(img_path)

        # Compute relative position (platform − drone)
        drone_pos = self.orchestrator.objects["drone"].getPos(mode="no_noise")
        platform_pos = self.orchestrator.objects["platform"].getPos(
            mode="no_noise"
        )
        rel = platform_pos - drone_pos  # numpy array length‑3

        # Append to CSV (thread‑safe)
        with self.csv_lock:
            with open(self.labels_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([img_name, f"{rel[0]:.6f}", f"{rel[1]:.6f}", f"{rel[2]:.6f}"])

    # ------------------------------------------------------------------
    # Override the internal _run loop to inject saving + capture limit
    # ------------------------------------------------------------------

    def _run(self):
        import glfw  # local import to avoid mandatory dependency during training

        if not glfw.init():
            raise RuntimeError("GLFW could not be initialized (needed for MJ rendering)")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # off‑screen window
        offscreen_window = glfw.create_window(self.width, self.height, "", None, None)
        glfw.make_context_current(offscreen_window)

        scene = mujoco.MjvScene(self.env.model, maxgeom=1000)
        context = mujoco.MjrContext(self.env.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        while self.running and self.counter < self.n_samples:
            # Lock camera to drone body (eye‑in‑hand)
            self.cam.lookat[:] = self.attached_body.sensors["gps"].getPos(
                mode="no_noise"
            )
            self.cam.distance = 0
            self.cam.azimuth = 0
            self.cam.elevation = -90

            mujoco.mjv_updateScene(
                self.env.model,
                self.env.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )

            rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            mujoco.mjr_render(
                mujoco.MjrRect(0, 0, self.width, self.height), scene, context
            )
            mujoco.mjr_readPixels(
                rgb_buffer,
                None,
                mujoco.MjrRect(0, 0, self.width, self.height),
                context,
            )
            rgb_image = np.flip(rgb_buffer, axis=0)

            # Save sample
            self._save_sample(rgb_image)
            self.counter += 1

            # Simple progress display
            if self.counter % 50 == 0 or self.counter == self.n_samples:
                print(f"Captured {self.counter}/{self.n_samples} frames", end="\r")

            # Respect FPS
            time.sleep(1.0 / self.update_rate)

        # Capture complete – stop simulation
        self.running = False
        self.orchestrator.ChangeLoopState(terminate=True)
        glfw.terminate()


# -----------------------------------------------------------------------------
# 2. TRAINING UTILITIES --------------------------------------------------------
# -----------------------------------------------------------------------------


class RelPosDataset(torch.utils.data.Dataset):
    """Image → 3‑D regression dataset using the CSV generated above."""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform or DefaultTransforms()
        self.samples: list[Tuple[Path, np.ndarray]] = []

        csv_path = self.root / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError("labels.csv not found – run in capture mode first")

        import pandas as pd

        df = pd.read_csv(csv_path)
        for row in df.itertuples(index=False):
            self.samples.append((self.root / row.filename, np.array([row.rel_x, row.rel_y, row.rel_z], dtype=np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(target)


class DefaultTransforms:
    """Lazy wrapper to avoid importing torchvision when not training."""

    def __init__(self):
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transforms(img)


class RelPosNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 3)

    def forward(self, x):
        return self.backbone(x)


# -----------------------------------------------------------------------------
# 3. HIGH‑LEVEL COMMANDS -------------------------------------------------------
# -----------------------------------------------------------------------------


def capture(samples: int, save_dir: str):
    """Run the MuJoCo sim in a background thread and capture *samples* frames."""
    print(f"Starting capture for {samples} frames …")

    orchestrator = Trainer()

    # 1️⃣ Simulation thread (runs the continuous step loop)
    sim_thread = threading.Thread(target=orchestrator.loop, daemon=True)
    sim_thread.start()

    # 2️⃣ Camera capture thread (headless)
    streamer = CaptureStreamer(
        orchestrator=orchestrator,
        attached_body=orchestrator.objects["drone"],
        save_dir=save_dir,
        n_samples=samples,
        update_rate=FPS,
    )
    streamer.start()

    # Wait until capture thread stops (it terminates the sim itself)
    streamer.thread.join()
    sim_thread.join()

    print("\nCapture complete. Data saved to", save_dir)


def train(data_dir: str, epochs: int, batch: int, lr: float):
    """Train the regression network using the captured dataset."""
    if torch is None:
        raise RuntimeError("PyTorch not installed – run: pip install torch torchvision")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RelPosDataset(data_dir)

    val_pct = 0.1
    val_size = int(len(ds) * val_pct)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=4)

    model = RelPosNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optim.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optim.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / train_size

        # Validation
        model.eval()
        running = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                running += loss.item() * imgs.size(0)
        val_loss = running / val_size

        print(f"Epoch {epoch:02d}/{epochs} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), Path(data_dir) / "relposnet_best.pth")
            print("  ↳ saved best model (val MSE ↓)")

    print("Training complete. Best val MSE:", best_val)
    print("Model saved to", Path(data_dir) / "relposnet_best.pth")


# -----------------------------------------------------------------------------
# 4. CLI ----------------------------------------------------------------------
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Drone rel‑pos dataset capture & trainer")
    parser.add_argument("--mode", choices=["capture", "train"], default="capture", help="capture data or train model")
    parser.add_argument("--samples", type=int, default=1000, help="number of frames to capture (capture mode)")
    parser.add_argument("--dataset", default="dataset", help="directory to save / read the dataset")
    parser.add_argument("--epochs", type=int, default=30, help="training epochs (train mode)")
    parser.add_argument("--batch", type=int, default=32, help="batch size (train mode)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (train mode)")

    args = parser.parse_args()

    if args.mode == "capture":
        capture(args.samples, args.dataset)
    else:
        train(args.dataset, args.epochs, args.batch, args.lr)


if __name__ == "__main__":
    main()
