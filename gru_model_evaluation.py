#!/usr/bin/env python
"""
Train a GRU-based model that maps a variable-length curve (sequence of (x, y) points)
to a 2D parameter vector w.

- Input sample: curve with shape (n_i, 2)
- Output: w with shape (2,)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple
from dataclasses import dataclass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import ImageToSCC as imscc
import utils.load_and_write as lw

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class TrainingConfig:
    hidden_dim: int = 128
    num_layers: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 10  # early stopping patience (epochs)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "./models"
    model_name: str = "curve_to_w_gru.pt"
    val_split: float = 0.2
    seed: int = 42


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


class CurveDataset(Dataset):
    """
    curves: list of np.ndarray or torch.Tensor, each of shape (n_i, 2)
    targets: np.ndarray or torch.Tensor of shape (N, 2)
    """

    def __init__(self, curves: List[np.ndarray], targets: np.ndarray, normalize: bool = True):
        assert len(curves) == len(targets), "curves and targets must have same length"
        self.curves = [torch.as_tensor(c, dtype=torch.float32) for c in curves]
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        pts = self.curves[idx]  # (n_i, 2)
        w = self.targets[idx]   # (2,)

        if self.normalize:
            pts = self._normalize_curve(pts)

        length = pts.shape[0]
        return pts, length, w

    @staticmethod
    def _normalize_curve(points: torch.Tensor) -> torch.Tensor:
        """
        Normalize curve to be roughly scale- and translation-invariant:
        - subtract centroid
        - divide by RMS distance to centroid (or 1 if zero to avoid divide-by-zero)
        """
        # points: (n, 2)
        centroid = points.mean(dim=0, keepdim=True)  # (1, 2)
        centered = points - centroid

        rms = torch.sqrt((centered ** 2).mean())
        if rms > 0:
            centered = centered / rms

        return centered


# ----------------------------------------------------------------------
# Collate function for variable-length sequences
# ----------------------------------------------------------------------


def collate_batch(batch: List[Tuple[torch.Tensor, int, torch.Tensor]]):
    """
    batch: list of (points, length, w)
    - points: (n_i, 2)
    - length: scalar
    - w: (2,)
    Returns:
        padded_points: (B, T_max, 2)
        lengths: (B,)
        ws: (B, 2)
    """
    points_list, lengths_list, ws_list = zip(*batch)
    lengths = torch.as_tensor(lengths_list, dtype=torch.long)  # (B,)
    batch_size = len(batch)
    max_len = max(lengths_list)

    padded_points = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)

    for i, pts in enumerate(points_list):
        n = pts.shape[0]
        padded_points[i, :n, :] = pts

    ws = torch.stack(ws_list, dim=0)  # (B, 2)
    return padded_points, lengths, ws


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------


class CurveToWModel(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.output_linear = nn.Linear(hidden_dim, 2)  # 2D w

    def forward(self, padded_points: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        padded_points: (B, T_max, 2)
        lengths: (B,)
        Returns:
            w_hat: (B, 2)
        """
        # Encode each (x, y) point
        x = self.input_linear(padded_points)  # (B, T_max, hidden_dim)

        # Pack for GRU to ignore padding steps
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, h_n = self.gru(packed)
        # h_n: (num_layers, B, hidden_dim)
        last_hidden = h_n[-1]  # final layer’s hidden state: (B, hidden_dim)

        w_hat = self.output_linear(last_hidden)  # (B, 2)
        return w_hat


# ----------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for padded_points, lengths, ws in loader:
        padded_points = padded_points.to(device)
        lengths = lengths.to(device)
        ws = ws.to(device)

        optimizer.zero_grad()
        w_hat = model(padded_points, lengths)
        loss = criterion(w_hat, ws)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for padded_points, lengths, ws in loader:
            padded_points = padded_points.to(device)
            lengths = lengths.to(device)
            ws = ws.to(device)

            w_hat = model(padded_points, lengths)
            loss = criterion(w_hat, ws)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def save_model(model: nn.Module, config: TrainingConfig):
    os.makedirs(config.model_dir, exist_ok=True)
    path = os.path.join(config.model_dir, config.model_name)
    torch.save({"model_state_dict": model.state_dict()}, path)
    print(f"Saved model to {path}")


def load_model(model: nn.Module, config: TrainingConfig, strict: bool = True) -> bool:
    path = os.path.join(config.model_dir, config.model_name)
    if not os.path.exists(path):
        print(f"No model found at {path}")
        return False

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    print(f"Loaded model from {path}")
    return True


# ----------------------------------------------------------------------
# Main training entrypoint
# ----------------------------------------------------------------------


def main(config: TrainingConfig):
    set_seed(config.seed)

    # TODO: Replace this with your real data loading
    # ------------------------------------------------------------------
    # Example dummy data: 1000 curves, random lengths between 20 and 100,
    # random (x, y) coordinates, and random targets w.
    # Replace this block with your actual curves & targets.
    # num_samples = 1000
    # min_len, max_len = 20, 100

    # curves: List[np.ndarray] = []
    # targets = np.zeros((num_samples, 2), dtype=np.float32)

    # for i in range(num_samples):
    #     n_i = np.random.randint(min_len, max_len + 1)
    #     # random curve
    #     pts = np.random.randn(n_i, 2).astype(np.float32)
    #     curves.append(pts)

    #     # example target: just for demonstration
    #     # In your case, this should be the true w_i*
    #     targets[i] = np.random.randn(2).astype(np.float32)
    # ------------------------------------------------------------------

    # Load curves data
    curves = load_curve_dataset_from_images('/Volumes/HOUSE MINI/IMAGENES/curves_500_5', 'coordinates_curves.txt', 'images')
    # curves = load_curve_dataset('/Users/zianfanti/IIMAS/images_databases/curves_200_5', 'coordinates_curves.txt', 'images')
    # curves = lw. load_pixelated_curves_from_txt_file('/Volumes/HOUSE MINI/IMAGENES/curves_500_5/pixel_curves')

    targets = load_targets_dataset('train/50iter_2500samples.csv')
    # targets = load_targets_dataset('/Users/zianfanti/IIMAS/Tree_Representation/src/back-forth/train/30itter_1000samples_separeted.csv')

    # Train/val split
    N = len(curves)
    indices = np.arange(N)
    np.random.shuffle(indices)

    split = int(N * (1 - config.val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_curves = [curves[i] for i in train_idx]
    val_curves = [curves[i] for i in val_idx]

    train_targets = targets[train_idx]
    val_targets = targets[val_idx]

    train_dataset = CurveDataset(train_curves, train_targets, normalize=True)
    val_dataset = CurveDataset(val_curves, val_targets, normalize=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    # Model, optimizer, loss
    model = CurveToWModel(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()

    # Training with early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("Start train")
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device
        )
        val_loss = evaluate(model, val_loader, criterion, config.device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-6:  # small tolerance
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_model(model, config)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Best validation loss: {best_val_loss:.6f}")



def load_curve_dataset_from_images(curves_dir_path, description_filename, image_folder):
    curves = []
    with open(os.path.join(curves_dir_path, description_filename), 'r', encoding='utf-8') as f:        
        for idx, line in enumerate(f):            
            match1 = re.search(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', line)               
            if match1:
                fname = match1.group(1)

                match2 = re.search(r'_(\d+)_X(\d+)', fname)
                if match2:
                    scale = float(match2.group(2)) / 10.
                    
                x = float(match1.group(2))
                y = float(match1.group(3))
                sp = (int(y), int(x))

                o_image = cv2.imread(os.path.join(curves_dir_path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)      

                k = 2
                branch = []                
                while type(treepath[k]) is tuple:
                    branch.append(treepath[k])                    
                    k += 1
                
                px = [point[0] for point in branch]
                py = [point[1] for point in branch]
                pixel_curve = np.column_stack([px, py])
                curves.append(np.array(pixel_curve))
    
    return curves




def load_targets_dataset(targets_filename):
    data = []

    with open(targets_filename, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]            
            num_points = float(parts[2])
            smooth = float(parts[3])
            data.append([num_points, smooth])

    return np.array(data)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--model_name", type=str, default="curve_to_w_gru.pt")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        model_dir=args.model_dir,
        model_name=args.model_name,
        val_split=args.val_split,
        seed=args.seed,
    )

    main(cfg)
