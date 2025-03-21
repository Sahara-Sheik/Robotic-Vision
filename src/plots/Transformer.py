import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# pip install timm
import timm

# --------------------------------------------------------
# 1) DATASET CLASS
# --------------------------------------------------------
class RobotImageDataset(Dataset):
    """
    scans data_dir for images and matching .json files.
    Returns (image_tensor, 6D tensor).
    Expected JSON format:
       {
         "rc-position-target": {
             "height": ...,
             "distance": ...,
             "rotation": ...,
             "wrist_angle": ...,
             "wrist_rotation": ...,
             "gripper": ...
         }
       }
    """

    def __init__(self, root_dir, image_size=224):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = []
        self._scan_directory()

    def _scan_directory(self):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        for subdir, _, files in os.walk(self.root_dir):
            for fname in files:
                if fname.lower().endswith(exts):
                    img_path = os.path.join(subdir, fname)
                    base_name, _ = os.path.splitext(fname)
                    cleaned_base_name = base_name.replace("_dev2", "")
                    json_path = os.path.join(subdir, cleaned_base_name + ".json")
                    if os.path.isfile(json_path):
                        # Load JSON
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        if "rc-position-target" not in data:
                            continue
                        # Extract 6D
                        pos = data["rc-position-target"]
                        sixd = [
                            float(pos["height"]),
                            float(pos["distance"]),
                            float(pos["rotation"]),       # or rename to 'heading'
                            float(pos["wrist_angle"]),
                            float(pos["wrist_rotation"]),
                            float(pos["gripper"])
                        ]
                        self.samples.append((img_path, sixd))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sixd = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise RuntimeError(f"Could not load image {img_path}")
        # Convert BGR->RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Resize to (224,224)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        # Convert to float32, scale [0..1]
        rgb_tensor = torch.from_numpy(rgb).float() / 255.0
        # (H, W, C) => (C, H, W)
        rgb_tensor = rgb_tensor.permute(2, 0, 1)

        sixd_tensor = torch.tensor(sixd, dtype=torch.float32)
        return rgb_tensor, sixd_tensor

# --------------------------------------------------------
# 2) VIT BACKBONE + REGRESSION HEAD
# --------------------------------------------------------
class ViTRegressor(nn.Module):
    """
    Wraps a timm Vision Transformer backbone, replacing the classification head
    with a small MLP for 6D regression.
    """
    def __init__(self, backbone='vit_base_patch16_224', pretrained=True, out_dim=6):
        super().__init__()
        # Load timm model, remove its classification head
        self.vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # num_classes=0 => remove default classifier => output is a 6-D feature vector

        # Inspect feature dim for chosen backbone
        # For 'vit_base_patch16_224', this is typically 768
        feature_dim = self.vit.num_features

        # Add a small MLP head for regression
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        # vit(...) => (B, feature_dim)
        feats = self.vit(x)
        out = self.mlp_head(feats)   # (B, 6)
        return out

# --------------------------------------------------------
# 3) MAIN TRAINING SCRIPT
# --------------------------------------------------------
def main():
    root_dir =  "/home/ssheikholeslami/BerryPicker2/BlueBerry"  # <-- update if we change to a central exoeriment
    dataset = RobotImageDataset(root_dir, image_size=224)
    print("Found samples:", len(dataset))
    if len(dataset) == 0:
        return

    # Convert entire dataset to memory for simple splitting
    all_imgs = []
    all_6d   = []
    for i in range(len(dataset)):
        img_t, sixd_t = dataset[i]
        all_imgs.append(img_t.numpy())
        all_6d.append(sixd_t.numpy())
    all_imgs = np.array(all_imgs, dtype=np.float32)
    all_6d   = np.array(all_6d,   dtype=np.float32)

    # Train/test split
    train_imgs, test_imgs, train_6d, test_6d = train_test_split(
        all_imgs, all_6d, test_size=0.1, random_state=42
    )
    print("Train size:", len(train_imgs), "Test size:", len(test_imgs))

    # Build PyTorch Datasets
    class ArrayDataset(Dataset):
        print("creating data set")
        def __init__(self, X, Y):
            self.X = torch.from_numpy(X)
            self.Y = torch.from_numpy(Y)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    print("spilitting data set")
    train_ds = ArrayDataset(train_imgs, train_6d)
    test_ds  = ArrayDataset(test_imgs,  test_6d)

    # DataLoaders
    print("data loaders")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False)

    # Create model + optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Training on GPU")
    else:
        print("Training on CPU")
    model = ViTRegressor(
        backbone='vit_base_patch16_224',  # or 'vit_small_patch16_224', etc.
        pretrained=True,
        out_dim=6
    ).to(device)

    print("model created")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("lets train")
    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, y6d in train_loader:
            imgs = imgs.to(device)
            y6d  = y6d.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, y6d)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(imgs)

        epoch_loss = running_loss / len(train_ds)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f}")

    # Evaluate on test set
    print("lets evaluate")
    model.eval()
    all_preds = []
    all_truth = []
    with torch.no_grad():
        for imgs, y6d in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            all_preds.append(preds.cpu().numpy())
            all_truth.append(y6d.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_truth = np.concatenate(all_truth, axis=0)

    # MSE/MAE per dimension
    mse = np.mean((all_preds - all_truth)**2, axis=0)
    mae = np.mean(np.abs(all_preds - all_truth), axis=0)
    print("MSE per dimension:", mse)
    print("MAE per dimension:", mae)

    # Plot a subset of predictions vs. ground truth

# Suppose `preds_all` are your model predictions of shape (N, 6)
# and `ytrue_all` are the ground-truth 6D of the same shape.
# param_names are the 6 parameters you want to plot.
    param_names = ["height", "distance", "heading", "wrist_angle", "wrist_rotation", "gripper"]

    N_to_plot = min(50, len(all_preds))  # or however many test samples you want to show
    gt_slice  = all_truth[:N_to_plot]
    pr_slice  = all_preds[:N_to_plot]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for i, pname in enumerate(param_names):
        axes[i].plot(gt_slice[:, i], label="ground truth", color='blue')
        axes[i].plot(pr_slice[:, i], label="prediction", color='orange')
        axes[i].set_title(pname)
        if i == 0:
            axes[i].legend(loc='upper right')

    plt.tight_layout()

    # Save plot in your current directory (or specify any path you'd like).
    plt.savefig("/home/ssheikholeslami/BerryPicker/src/plots/results_plot.png", dpi=300)
    plt.close()  # Close the figure so it does not display

if __name__ == "__main__":
    main()
