import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from network.decom import Decom
from network.Math_Module import P, Q
from utils import define_modelR, define_modelL

# ============================================================
# Config
# ============================================================

# INPUT_DIR = "./diff_combinations/003expNorm"
# OUTPUT_DIR = "./diff_combinations/003expNorm_T5"
# INPUT_DIR = "./panel_data_original/train_normal_G4"
# OUTPUT_DIR = "./panel_data_T5/train_normal_G4"

INPUT_DIR = "./Tesi_example"
OUTPUT_DIR = "./Tesi_example_results"


DECOM_LOW_PATH = "./pretrained_model/decom/decom_low_light.pth"
UNFOLDING_PATH = "./pretrained_model/unfolding/unfolding_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Create Output Folder
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# Save Helper
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
# Load Decom Low
# ============================================================

print("Loading Decom Low...")

model_Decom_low = Decom().to(device)

ckpt_low = torch.load(
    DECOM_LOW_PATH,
    map_location=device
)

model_Decom_low.load_state_dict(
    ckpt_low["state_dict"]["model_R"]
)

model_Decom_low.eval()

print("Decom Low Loaded")

# ============================================================
# Load Unfolding
# ============================================================

print("Loading Unfolding Model...")

unfold_ckpt = torch.load(
    UNFOLDING_PATH,
    map_location=device,
    weights_only=False
)

opts = unfold_ckpt["opts"]

model_R = define_modelR(
    opts
).to(device)

model_L = define_modelL(
    opts
).to(device)

model_R.load_state_dict(
    unfold_ckpt["state_dict"]["model_R"]
)

model_L.load_state_dict(
    unfold_ckpt["state_dict"]["model_L"]
)

model_R.eval()
model_L.eval()

print("Unfolding Model Loaded")

# ============================================================
# Math Modules
# ============================================================

P_module = P()
Q_module = Q()

# ============================================================
# Find Images
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
# Process
# ============================================================

with torch.no_grad():

    for image_name in image_list:

        print(f"\nProcessing: {image_name}")

        image_path = os.path.join(
            INPUT_DIR,
            image_name
        )

        img = Image.open(
            image_path
        ).convert("RGB")

        img_tensor = transform(
            img
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Initialize
        # ----------------------------------------------------

        P_cur, Q_cur = model_Decom_low(
            img_tensor
        )

        R = None
        L = None

        # ----------------------------------------------------
        # T = 5
        # ----------------------------------------------------

        for t in range(5):

            if t > 0:

                gamma = (
                    opts.gamma
                    + opts.Roffset * t
                )

                lamda = (
                    opts.lamda
                    + opts.Loffset * t
                )

                P_cur = P_module(
                    I=img_tensor,
                    Q=Q_cur,
                    R=R,
                    gamma=gamma
                )

                Q_cur = Q_module(
                    I=img_tensor,
                    P=P_cur,
                    L=L,
                    lamda=lamda
                )

            R = model_R(
                r=P_cur,
                l=Q_cur
            )

            L = model_L(
                l=Q_cur
            )

        # ----------------------------------------------------
        # Save T5
        # ----------------------------------------------------

        base_name = os.path.splitext(
            image_name
        )[0]

        # save_path = os.path.join(
        #     OUTPUT_DIR,
        #     f"{base_name}_expNorm_T5.png"
        # )

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{base_name}_T5.png"
        )


        save_tensor_image(
            R,
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