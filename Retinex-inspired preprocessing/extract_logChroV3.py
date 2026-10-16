import os
import cv2
import numpy as np
from PIL import Image



# ============================================================
# 1. Log-Chromaticity 纹理提取 (完全保留你的原始算法)
# ============================================================

def extract_texture_H(img, sigma=25, eps=0.025):
    """
    H = logY - Gaussian(logY)

    Parameters
    ----------
    img : float32 RGB image in [0,1]
    sigma : Gaussian blur sigma

    Returns
    -------
    H : float32 texture map
    """
    img = np.clip(img, eps, 1.0)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Luminance (BT.709)
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    logY = np.log(np.clip(Y, eps, 1.0))

    smooth = cv2.GaussianBlur(
        logY,
        ksize=(0, 0),
        sigmaX=sigma
    )

    H = logY - smooth

    return H.astype(np.float32)


# ============================================================
# 2. 图像保存函数 (保留原始动态 Min-Max，输出 3 通道 RGB)
# ============================================================

def save_H_image_for_model(H, save_path):
    """
    保留最原始的自适应 Min-Max 归一化逻辑，确保包含大黑孔、线缆等
    复杂结构的大图时灰度过渡平滑自然，绝不爆扣或二值化。

    改进：直接保存为 3 通道 RGB 图像，供 PatchCore / ResNet 无缝读取。
    """
    h_min = H.min()
    h_max = H.max()

    if h_max - h_min < 1e-8:
        H_vis = np.zeros_like(H, dtype=np.uint8)
    else:
        # 原始代码的自适应动态映射
        H_vis = ((H - h_min) / (h_max - h_min) * 255.0).astype(np.uint8)

    # 将 1 通道灰度扩展为 3 通道 RGB (Shape: H x W x 3)
    H_rgb = np.stack([H_vis, H_vis, H_vis], axis=-1)

    # 保存为标准 RGB 模式图片
    Image.fromarray(H_rgb, mode="RGB").save(save_path)


# ============================================================
# 3. 参数配置
# ============================================================

# # INPUT_DIR = "./diff_combinations/006T5_expNorm"
# # OUTPUT_DIR = "./diff_combinations/006T5_expNorm_LogChro"
# INPUT_DIR = "./panel_data_T0high/train_normal_G4"
# OUTPUT_DIR = "./panel_data_T0high_log/train_normal_G4"

INPUT_DIR = "./Tesi_example"
OUTPUT_DIR = "./Tesi_example_results"


SIGMA = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 4. 扫描文件
# ============================================================

image_list = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])

print(f"Found {len(image_list)} image files")

# ============================================================
# 5. 循环处理并保存
# ============================================================

for image_name in image_list:
    image_path = os.path.join(INPUT_DIR, image_name)
    print(f"Processing: {image_name}")

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0

    # 提取纹理
    H = extract_texture_H(img_np, sigma=SIGMA)

    # 保存为模型专用的 3 通道 RGB 格式（文件名保持与原始图完全一致）
    save_path = os.path.join(OUTPUT_DIR, image_name)
    save_H_image_for_model(H, save_path)

    print(f"Saved -> {save_path}")

# ============================================================
# 6. 完成提示
# ============================================================

print("\n================================")
print("Finished!")
print("Output Folder:")
print(os.path.abspath(OUTPUT_DIR))
print("================================")