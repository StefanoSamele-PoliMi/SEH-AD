import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from network.decom import Decom

# ============================================================
# Config
# ============================================================

# # INPUT_DIR = "./diff_combinations/004expNorm"
# INPUT_DIR = "./panel_data_original/train_normal_G4"
#
# # OUTPUT_DIR = "./diff_combinations/004expNorm_T0low"
# OUTPUT_DIR = "./panel_data_T0low/train_normal_G4"

INPUT_DIR = "./Tesi_example"
OUTPUT_DIR = "./Tesi_example_results"

DECOM_LOW_PATH = "./pretrained_model/decom/decom_low_light.pth"

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
# Load Decom Low Model
# ============================================================

print("Loading Decom Low Model...")

model_Decom_low = Decom().to(device)

ckpt = torch.load(
    DECOM_LOW_PATH,
    map_location=device
)

model_Decom_low.load_state_dict(
    ckpt["state_dict"]["model_R"]
)

model_Decom_low.eval()

print("Decom Low Model Loaded")

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
        # Decomposition
        # ----------------------------------------------------

        R_low, L_low = model_Decom_low(
            img_tensor
        )

        # ----------------------------------------------------
        # Save Reflectance (T0_low)
        # ----------------------------------------------------

        base_name = os.path.splitext(
            image_name
        )[0]

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{base_name}_T0low.png"
        )
        # save_path = os.path.join(
        #     OUTPUT_DIR,
        #     f"{base_name}.png"
        # )

        save_tensor_image(
            R_low,
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