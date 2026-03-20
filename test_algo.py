import cv2
import numpy as np
import matplotlib.pyplot as plt
from algorithms import lut, slicing, mapping
import os

# Create a test grayscale image (gradient + some shapes)
def make_test_image():
    img = np.zeros((256, 256), dtype=np.uint8)
    # Gradient background
    for i in range(256):
        img[i, :] = i
    # Add a circle and rectangle for structure
    cv2.circle(img, (128, 128), 80, 200, -1)
    cv2.rectangle(img, (40, 40), (100, 100), 50, -1)
    return img

gray = make_test_image()

results = {
    "Original grayscale":      cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
    "LUT: jet":                cv2.cvtColor(lut.apply_lut(gray, "jet"),             cv2.COLOR_BGR2RGB),
    "LUT: plasma":             cv2.cvtColor(lut.apply_lut(gray, "plasma"),          cv2.COLOR_BGR2RGB),
    "LUT: custom (B-G-R)":     cv2.cvtColor(lut.apply_custom_lut(gray,
                                    lut.build_custom_lut()), cv2.COLOR_BGR2RGB),
    "Slicing: 4 levels":       cv2.cvtColor(slicing.apply_level_slicing(gray, 4),   cv2.COLOR_BGR2RGB),
    "Slicing: 6 levels":       cv2.cvtColor(slicing.apply_level_slicing(gray, 6),   cv2.COLOR_BGR2RGB),
    "Slicing: bg preserved":   cv2.cvtColor(slicing.apply_level_slicing(gray, 4,
                                    background=True), cv2.COLOR_BGR2RGB),
    "Mapping: sine f=1.0":     cv2.cvtColor(mapping.apply_sin_mapping(gray, 1.0),   cv2.COLOR_BGR2RGB),
    "Mapping: density":        cv2.cvtColor(mapping.apply_density_mapping(gray),    cv2.COLOR_BGR2RGB),
    "Mapping: gamma+plasma":   cv2.cvtColor(mapping.apply_gamma_mapped(gray, 2.0),  cv2.COLOR_BGR2RGB),
}

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, (title, img) in zip(axes.flatten(), results.items()):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/algorithm_test.png", dpi=150)
print("Saved → outputs/algorithm_test.png")