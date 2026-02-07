"""
Single-Backbone Parallel Multi-System Training (GPU-Optimized)
===============================================================
Paper: "Learning Laws for Deep CNNs with Guaranteed Convergence"

Architecture: 1 model7 backbone + 8 auxiliary classifiers
Speed: FASTER than original (1 batched forward vs 8 redundant)
Correctness: SAME as original (batched forward + per-sample update)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ running on", DEVICE)

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import time
import math

from helper1 import train_all_systems, DEVICE_
import my_module1 as mm
from my_module1 import build_multi_system, evaluate_all_systems, SYSTEMS_CONFIG

# ===== Configuration =====
BATCH_SIZE = 64
N_CLASSES = 10
N_EPOCHS = 200

# Adjustable ReLU schedule (paper §4.3)
a_start = 0.25
a_end = 0.01
a_epochs = 30

learnrate = 0.001

PRINT_EVERY = 100          # print batch progress every N batches
SAVE_EVERY  = 10           # periodic checkpoint interval (epochs)

# ===== Data Loading =====
FOLDER_e2e = mm.FOLDER_e2e
if not os.path.exists(FOLDER_e2e):
    os.makedirs(FOLDER_e2e, exist_ok=True)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.SVHN(root='./data', split='train',
                              transform=train_transform, download=True)
val_dataset = datasets.SVHN(root='./data', split='test',
                            transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f'Data: {len(train_dataset)} train, {len(val_dataset)} val (SVHN test split)')

# ===== Device sync =====
mm.DEVICE = DEVICE
DEVICE_[0] = DEVICE

# ===== Build backbone + 8 auxiliary classifiers =====
print('\n===== Building Multi-System Backbone =====')
model, aux_classifiers = build_multi_system(N_CLASSES, train_loader)
model = model.float()
print(f'Backbone: {len(model.layers)} layers, {len(SYSTEMS_CONFIG)} systems\n')

# ===== Training =====
print('=' * 50)
print(' Training Start')
print('=' * 50)

a_slope = a_start
best_val = [0.0] * len(SYSTEMS_CONFIG)
best_avg_val = 0.0
t0 = time.time()
n_batches = len(train_loader)

for epoch in range(N_EPOCHS):
    epoch_t0 = time.time()

    # --- Adjustable ReLU schedule ---
    if epoch < a_epochs:
        a_slope = a_start + (a_end - a_start) * epoch / a_epochs
    else:
        a_slope = a_end

    # --- Batch loop with progress ---
    epoch_e_norms = None
    dataiter = iter(train_loader)
    for batch_idx in range(n_batches):
        try:
            x, y = next(dataiter)
        except StopIteration:
            dataiter = iter(train_loader)
            x, y = next(dataiter)

        aux_classifiers, e_norms = train_all_systems(
            model, aux_classifiers, SYSTEMS_CONFIG,
            epoch_no=epoch, batch_no=batch_idx,
            x=x, y=y, n_classes=N_CLASSES,
            slope=a_slope, gain=learnrate, auto=True
        )
        epoch_e_norms = e_norms

        if (batch_idx + 1) % PRINT_EVERY == 0 or batch_idx == n_batches - 1:
            e_str = ' '.join(f'{e:.3f}' for e in e_norms)
            print(f'  Epoch {epoch} [{batch_idx+1}/{n_batches}]  '
                  f'e_norm=[{e_str}]  a={a_slope:.4f}')

    epoch_time = time.time() - epoch_t0

    # --- Evaluate all systems ---
    train_accs, val_accs = evaluate_all_systems(
        model, aux_classifiers, SYSTEMS_CONFIG, train_loader, val_loader)

    # --- Print epoch summary ---
    elapsed = time.time() - t0
    print(f'\n{"─"*50}')
    print(f'Epoch {epoch}/{N_EPOCHS}  │  a={a_slope:.4f}  │  '
          f'epoch={epoch_time:.1f}s  total={elapsed:.1f}s')
    for j in range(len(SYSTEMS_CONFIG)):
        improved = val_accs[j] > best_val[j]
        star = ' ★' if improved else ''
        if improved:
            best_val[j] = val_accs[j]
        print(f'  S_{j}: train={train_accs[j]:.2f}%  '
              f'val={val_accs[j]:.2f}%  '
              f'e={epoch_e_norms[j]:.4f}{star}')

    avg_val = sum(val_accs) / len(val_accs)
    print(f'  Avg val: {avg_val:.2f}%  │  '
          f'Best per-sys: {[f"{a:.1f}" for a in best_val]}')

    # --- Save best model (by average val accuracy) ---
    if avg_val > best_avg_val:
        best_avg_val = avg_val
        ckpt_path = os.path.join(FOLDER_e2e, 'best_model.pt')
        torch.save({
            'epoch': epoch, 'model': model,
            'aux': [a.cpu().clone() for a in aux_classifiers],
            'best_val': best_val, 'avg_val': avg_val,
            'a_slope': a_slope,
        }, ckpt_path)
        print(f'  💾 Best model saved → {ckpt_path}  (avg_val={avg_val:.2f}%)')

    # --- Periodic checkpoint ---
    if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS - 1:
        ckpt_path = os.path.join(FOLDER_e2e, f'ckpt_e{epoch}.pt')
        torch.save({
            'epoch': epoch, 'model': model,
            'aux': [a.cpu().clone() for a in aux_classifiers],
            'best_val': best_val, 'a_slope': a_slope,
        }, ckpt_path)
        print(f'  💾 Checkpoint saved → {ckpt_path}')

    print(f'{"─"*50}\n')

print('=' * 50)
print(f' Done. Total time: {time.time()-t0:.1f}s')
print(f' Best avg val: {best_avg_val:.2f}%')
print(f' Best per-system: {best_val}')
print('=' * 50)
