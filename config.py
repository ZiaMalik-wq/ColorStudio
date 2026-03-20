import os

# ── Paths ──────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "raw")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")

# ── Dataset ────────────────────────────────────────────────
DATASET        = "places365"   # options: "places365", "stl10"
IMAGE_SIZE     = 128           # resize all images to this
BATCH_SIZE     = 8
NUM_WORKERS    = 2

# ── Training ───────────────────────────────────────────────
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 30
DEVICE         = "cpu"        # change to "cpu" if no GPU

# ── Color space ────────────────────────────────────────────
# We work in LAB space: L = grayscale input, AB = color output
L_MEAN         = 50.0
AB_NORM        = 110.0         # normalize AB channels to [-1, 1]
