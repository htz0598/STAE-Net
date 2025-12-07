import os
import re
import csv
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Utils: reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
def parse_idx_and_label(filename: str):
    """
    Parse idx and label from filenames like:
      'train_idx000123_label-Normal.png'
      'test_idx000045_label-Deauth.png'
    """
    m_idx = re.search(r"idx(\d+)", filename)
    if m_idx:
        idx = int(m_idx.group(1))
    else:
        raise ValueError(f"Cannot parse idx from filename: {filename}")

    m_label = re.search(r"label-([^\.]+)", filename)
    if m_label:
        label = m_label.group(1)
    else:
        label = "Unknown"

    return idx, label


class ImageFolderDataset(Dataset):
    """
    Dataset that loads RGB images from a folder,
    and returns (image_tensor, idx, label).
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        files = [f for f in os.listdir(root_dir) if f.lower().endswith(".png")]
        if not files:
            raise RuntimeError(f"No PNG files found in {root_dir}")

        file_info = []
        for fname in files:
            idx, label = parse_idx_and_label(fname)
            file_info.append((idx, label, fname))

        # sort by idx to preserve original order
        file_info.sort(key=lambda x: x[0])

        self.indices = [fi[0] for fi in file_info]
        self.labels = [fi[1] for fi in file_info]
        self.paths = [os.path.join(root_dir, fi[2]) for fi in file_info]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        idx = self.indices[i]
        label = self.labels[i]

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
        # H, W, C -> C, H, W
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr)  # shape: (3, H, W)

        return tensor, idx, label


# =========================
# Model: SE + Residual + U-Net style
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = self.avg_pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class ResAE(nn.Module):
    """
    Residual Autoencoder with SE and U-Net-style skip connections.
    Input / output: (B, 3, 256, 256), values in [0,1].
    """

    def __init__(self, reduction=8):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(3, 64, stride=1, reduction=reduction)
        self.down1 = ResidualBlock(64, 64, stride=2, reduction=reduction)   # 256 -> 128

        self.enc2 = ResidualBlock(64, 128, stride=1, reduction=reduction)
        self.down2 = ResidualBlock(128, 128, stride=2, reduction=reduction) # 128 -> 64

        self.enc3 = ResidualBlock(128, 256, stride=1, reduction=reduction)
        self.down3 = ResidualBlock(256, 256, stride=2, reduction=reduction) # 64 -> 32

        self.enc4 = ResidualBlock(256, 512, stride=1, reduction=reduction)
        self.down4 = ResidualBlock(512, 512, stride=2, reduction=reduction) # 32 -> 16

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = ResidualBlock(512 + 512, 512, stride=1, reduction=reduction)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ResidualBlock(512 + 256, 256, stride=1, reduction=reduction)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResidualBlock(256 + 128, 128, stride=1, reduction=reduction)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ResidualBlock(128 + 64, 64, stride=1, reduction=reduction)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        # final activation = sigmoid to map back to [0,1]
        # (we will apply it in forward)

    def forward(self, x):
        # Encoder with skips
        x1 = self.enc1(x)              # (B, 64, 256, 256)
        d1 = self.down1(x1)            # (B, 64, 128, 128)

        x2 = self.enc2(d1)             # (B, 128, 128, 128)
        d2 = self.down2(x2)            # (B, 128, 64, 64)

        x3 = self.enc3(d2)             # (B, 256, 64, 64)
        d3 = self.down3(x3)            # (B, 256, 32, 32)

        x4 = self.enc4(d3)             # (B, 512, 32, 32)
        d4 = self.down4(x4)            # (B, 512, 16, 16)

        # Decoder: upsample + concat skip
        u4 = self.up4(d4)              # (B, 512, 32, 32)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.dec4(u4)             # (B, 512, 32, 32)

        u3 = self.up3(u4)              # (B, 512, 64, 64)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3(u3)             # (B, 256, 64, 64)

        u2 = self.up2(u3)              # (B, 256, 128, 128)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)             # (B, 128, 128, 128)

        u1 = self.up1(u2)              # (B, 128, 256, 256)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1(u1)             # (B, 64, 256, 256)

        out = self.out_conv(u1)        # (B, 3, 256, 256)
        out = torch.sigmoid(out)       # map to [0,1]
        return out


# =========================
# Training & Evaluation
# =========================
def train_resae(
    train_root="train_images",
    test_root="test_images",
    model_path="resae_awid3.pth",
    scores_train_csv="resae_scores_train.csv",
    scores_test_csv="resae_scores_test.csv",
    batch_size=16,
    num_epochs=60,
    lr=1e-4,
    device=None,
):
    set_seed(42)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading train dataset...")
    train_dataset = ImageFolderDataset(train_root)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    print("Loading test dataset...")
    test_dataset = ImageFolderDataset(test_root)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print("Building ResAE model...")
    model = ResAE(reduction=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # as in the paper

    # ---------- Training ----------
    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for imgs, _, _ in train_loader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            recons = model(imgs)
            loss = criterion(recons, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch}/{num_epochs}] - Train L1 Loss: {epoch_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Saved ResAE model to {model_path}")

    # ---------- Compute reconstruction errors on train & test ----------
    def compute_scores(dataloader, split_name):
        model.eval()
        all_indices = []
        all_labels = []
        all_errors = []

        with torch.no_grad():
            for imgs, idxs, labels in dataloader:
                imgs = imgs.to(device)
                recons = model(imgs)
                # mean absolute error per sample
                err = torch.mean(torch.abs(recons - imgs), dim=(1, 2, 3))  # shape (B,)

                all_indices.extend(idxs.numpy().tolist())
                all_labels.extend(list(labels))
                all_errors.extend(err.cpu().numpy().tolist())

        print(f"{split_name}: computed scores for {len(all_indices)} samples.")
        return all_indices, all_labels, all_errors

    print("Computing reconstruction errors on TRAIN set...")
    train_indices, train_labels, train_errors = compute_scores(train_loader, "Train")

    print("Computing reconstruction errors on TEST set...")
    test_indices, test_labels, test_errors = compute_scores(test_loader, "Test")

    # ---------- Save CSV ----------
    def save_scores_csv(path, indices, labels, errors):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "label", "E_img"])
            for i, lab, e in zip(indices, labels, errors):
                writer.writerow([i, lab, e])
        print(f"Saved scores to: {path}")

    save_scores_csv(scores_train_csv, train_indices, train_labels, train_errors)
    save_scores_csv(scores_test_csv, test_indices, test_labels, test_errors)


def main():
    train_resae(
        train_root="train_images",
        test_root="test_images",
        model_path="resae_awid3.pth",
        scores_train_csv="resae_scores_train.csv",
        scores_test_csv="resae_scores_test.csv",
        batch_size=16,
        num_epochs=60,
        lr=1e-4,
    )


if __name__ == "__main__":
    main()
