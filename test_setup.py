# test_setup.py  — run this to verify everything works
import torch
import cv2
import numpy as np
from skimage import color
from dataset import ColorizationDataset

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ OpenCV:", cv2.__version__)

print("\nLoading 2 samples from Places365 val split (will download on first run)...")
ds = ColorizationDataset(split="val")
L, AB = ds[0]
print(f"✓ L channel shape:  {L.shape}   (should be [1, 256, 256])")
print(f"✓ AB channels shape: {AB.shape}  (should be [2, 256, 256])")
print(f"✓ L range:  [{L.min():.2f}, {L.max():.2f}]  (should be near [-1, 1])")
print(f"✓ AB range: [{AB.min():.2f}, {AB.max():.2f}] (should be near [-1, 1])")
print("\nSetup complete! Ready for Phase 3.")