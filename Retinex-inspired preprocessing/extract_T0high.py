import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from network.decom import Decom

# ============================================================
# Config
# ============================================================

# INPUT_DIR = "./diff_combinations/original_imgs"
# OUTPUT_DIR = "./diff_combinations/T0high"
# INPUT_DIR = "./diff_combinations/006T5_expNorm"
# OUTPUT_DIR = "./diff_combinations/006T5_expNorm_LogChro"
# INPUT_DIR = "./panel_data_original/train_normal_G4"
# OUTPUT_DIR = "./panel_data_T0high/train_normal_G4"

INPUT_DIR = "./Tesi_example" # this should be the output of docker
OUTPUT_DIR = "./Tesi_example_results"

DECOM_HIGH_PATH = "./pretrained_model/decom/decom_high_light.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Create Output Folder
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# Save Tensor Helper
# ============================================================

def save_tensor_image(tensor, save_path):

    tensor = tensor.squeeze(0)

    tensor = torch.clamp(
        tensor,
        0,
        1
    )

    img = transforms.ToPILImage()(
        tensor.cpu()
    )

    img.save(save_path)

# ============================================================
# Load Decom High Model
# ============================================================

print("Loading Decom High Model...")

model_Decom_high = Decom().to(device)

ckpt = torch.load(
    DECOM_HIGH_PATH,
    map_location=device
)

model_Decom_high.load_state_dict(
    ckpt["state_dict"]["model_R"]
)

model_Decom_high.eval()

print("Decom High Model Loaded")

# ============================================================
# Find All PNG Images
# ============================================================

image_list = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".png")
])

print()
print(f"Found {len(image_list)} PNG files")

# ============================================================
# Transform
# ============================================================

transform = transforms.ToTensor()

# ============================================================
# Process Images
# ============================================================

with torch.no_grad():

    for image_name in image_list:

        image_path = os.path.join(
            INPUT_DIR,
            image_name
        )

        print(f"\nProcessing: {image_name}")

        img = Image.open(
            image_path
        ).convert("RGB")

        img_tensor = transform(
            img
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Decomposition High
        # ----------------------------------------------------

        R_high, L_high = model_Decom_high(
            img_tensor
        )

        # ----------------------------------------------------
        # Save Reflectance (T0_high)
        # ----------------------------------------------------

        base_name = os.path.splitext(
            image_name
        )[0]

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{base_name}_T0normal.png"
        )

        save_tensor_image(
            R_high,
            save_path
        )

        print(
            f"Saved -> {save_path}"
        )

# ============================================================
# Finish
# ============================================================

print("\n======================================")
print("Finished!")
print("Results saved to:")
print(os.path.abspath(OUTPUT_DIR))
print("======================================")