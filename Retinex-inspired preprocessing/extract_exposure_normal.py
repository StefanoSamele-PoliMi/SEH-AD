import os
import numpy as np
from PIL import Image

# ============================================================
# Exposure Normalization
# ============================================================

def exposure_normalize(
        img,
        mask=None,
        target_median=0.4,
        eps=1e-6):
    """
    Robust global exposure normalization
    using luminance median.
    """

    if mask is None:
        mask = np.ones(
            img.shape[:2],
            dtype=bool
        )

    Y = (
        0.2126 * img[:, :, 0]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 2]
    )

    valid = (
        mask
        & (Y < 0.98)
        & (Y > 0.02)
    )

    med_Y = np.median(
        Y[valid]
    )

    scale = (
        target_median
        / (med_Y + eps)
    )

    out = img * scale

    return np.clip(
        out,
        0.0,
        1.0
    )

# ============================================================
# Save Image
# ============================================================

def save_rgb_image(
        img_float,
        save_path):
    """
    float[0,1] -> PNG
    """

    img_uint8 = (
        img_float * 255
    ).astype(np.uint8)

    Image.fromarray(
        img_uint8
    ).save(save_path)

# ============================================================
# Config
# ============================================================

# INPUT_DIR = "./diff_combinations/006T5"
# OUTPUT_DIR = "./diff_combinations/006T5_expNorm"

INPUT_DIR = "./Tesi_example"
OUTPUT_DIR = "./Tesi_example_results"

TARGET_MEDIAN = 0.4

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# Scan Images
# ============================================================

image_list = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".png")
])

print(
    f"Found {len(image_list)} PNG files"
)

# ============================================================
# Process
# ============================================================

for image_name in image_list:

    image_path = os.path.join(
        INPUT_DIR,
        image_name
    )

    print(
        f"Processing: {image_name}"
    )

    # ------------------------------------------
    # Read Image
    # ------------------------------------------

    img = Image.open(
        image_path
    ).convert("RGB")

    img_np = (
        np.array(img)
        .astype(np.float32)
        / 255.0
    )

    # ------------------------------------------
    # Exposure Normalize
    # ------------------------------------------

    img_norm = exposure_normalize(
        img_np,
        target_median=TARGET_MEDIAN
    )

    # ------------------------------------------
    # Save
    # ------------------------------------------

    base_name = os.path.splitext(
        image_name
    )[0]

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}.png"
    )

    save_rgb_image(
        img_norm,
        save_path
    )

    print(
        f"Saved -> {save_path}"
    )

# ============================================================
# Finish
# ============================================================

print("\n================================")
print("Finished!")
print("Output Folder:")
print(os.path.abspath(OUTPUT_DIR))
print("================================")