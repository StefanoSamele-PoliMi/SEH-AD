#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, List, Tuple
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning as L

from anomalib.data.image.folder import Folder
from anomalib import TaskType
from anomalib.engine import Engine
from anomalib.metrics import F1AdaptiveThreshold, ManualThreshold, AUROC, PRO
from anomalib.models import Patchcore
from anomalib.utils.post_processing import superimpose_anomaly_map, anomaly_map_to_color_map
from torchvision.transforms.v2 import Compose, RandomAdjustSharpness, RandomHorizontalFlip, Resize, Normalize, CenterCrop


def process_and_save_image(image_path, left_dir, crop_dimensions,resize_dims):
    """
    Process an image: center crop, then resize with appropriate interpolation.
    """
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        crop_width, crop_height = crop_dimensions

        # 1. Calculate cropping box
        left = (img_width - crop_width) // 2
        upper = (img_height - crop_height) // 2
        right = left + crop_width
        lower = upper + crop_height

        # 2. Perform center crop
        cropped_img = img.crop((left, upper, right, lower))

        # 3. Determine Interpolation Method
        # Check if "mask" is in the path (case-insensitive)
        if "mask" in image_path.lower():
            # For Masks: Use NEAREST to keep it binary (0 and 255 only)
            resample_mode = Image.Resampling.NEAREST
        else:
            # For Normal Images: Use LANCZOS for high quality
            resample_mode = Image.Resampling.LANCZOS

        # 4. Perform Resize (to 256x256)
        final_img = cropped_img.resize((resize_dims), resample=resample_mode)

        # 5. Save the image
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        final_img.save(os.path.join(left_dir, f"{name}_cropped{ext}"))


def uneq_process_images_in_directory(source_dir, output_base_dir, crop_dimensions,resize_dims):
    """
    Process all images in the source directory: center crop, split (25%-50%-25%), and save patches.
    """
    # # Define directories for patches
    # left_dir = os.path.join(output_base_dir, 'left_patches')
    # middle_dir = os.path.join(output_base_dir, 'middle_patches')
    # right_dir = os.path.join(output_base_dir, 'right_patches')
    target_dir = output_base_dir

    # # Create directories if they don't exist
    # os.makedirs(left_dir, exist_ok=True)
    # os.makedirs(middle_dir, exist_ok=True)
    # os.makedirs(right_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    # Process each image in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(source_dir, filename)
            process_and_save_image(image_path, target_dir, crop_dimensions,resize_dims)

# When called, it should take the raw dataset in `dataset_dir` and
# create a tiled dataset inside it.


def create_tiled_dataset(dataset_dir, dataset_dirs_op):
    # normal_dir_ip = os.path.join(dataset_dir, "panel/train")
    # normal_test_dir_ip = os.path.join(dataset_dir, "panel/test/normal")
    # abnormal_dir_ip = os.path.join(dataset_dir, "panel/test/anomaly")
    # mask_dir_ip = os.path.join(dataset_dir, "panel/masks_panel")
    normal_dir_ip = os.path.join(dataset_dir, "candy/train")
    normal_test_dir_ip = os.path.join(dataset_dir, "candy/test/normal")
    abnormal_dir_ip = os.path.join(dataset_dir, "candy/test/anomaly")
    mask_dir_ip = os.path.join(dataset_dir, "candy/masks_candy")

    # This parameters should be fixed and coherent with the predict_patched_per_partnorm.py script
    # crop_dimensions = (1236, 300)
    # resize_dimensions = (1280, 720)
    crop_dimensions = (768, 768)
    resize_dims=(256,256)

    uneq_process_images_in_directory(normal_dir_ip, dataset_dirs_op["normal_dir"], crop_dimensions,resize_dims)
    uneq_process_images_in_directory(normal_test_dir_ip, dataset_dirs_op["normal_test_dir"], crop_dimensions,resize_dims)
    uneq_process_images_in_directory(abnormal_dir_ip, dataset_dirs_op["abnormal_dir"], crop_dimensions,resize_dims)
    uneq_process_images_in_directory(mask_dir_ip, dataset_dirs_op["mask_dir"], crop_dimensions,resize_dims)

    print("Created tile dataset @ ", dataset_dirs_op)


def visualiser(sample_dict, sample_idx, min_value, max_value):
    """
    Visualize anomaly detection results for a single sample.

    Args:
        sample_dict (dict): Dictionary containing a batch of results.
        sample_idx (int): Index of the specific sample within the batch.
        min_value (float): Global minimum anomaly score used for normalization.
        max_value (float): Global maximum anomaly score used for normalization.
    """
    # 1. Initialize the figure and subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # 2. Handle image path and loading
    p = sample_dict["image_path"][sample_idx]
    # Handle cases where image_path might be a nested list
    actual_path = p[0] if isinstance(p, list) else p

    img = cv2.imread(actual_path)
    if img is None:
        print(f"warning: can not read image {actual_path}")
        return plt

    # Convert BGR to RGB and resize to 256x256 to align with anomaly map dimensions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))

    # 3. Process Anomaly Map
    anomaly_map = sample_dict["anomaly_maps"][sample_idx].cpu().numpy()

    # Normalization logic (added 1e-10 to prevent division by zero)
    anomaly_map_normalized = (anomaly_map - min_value) / (max_value - min_value + 1e-10)

    # 4. Generate Superimposed Heatmap
    # Both img_resized and anomaly_map_normalized are now 256x256, preventing dimension mismatch errors
    heat_map = superimpose_anomaly_map(
        anomaly_map=anomaly_map_normalized,
        image=img_resized,
        normalize=False
    )

    # 5. Process Predicted Mask (binary segmentation)
    pred_mask = sample_dict["pred_masks"][sample_idx].cpu().numpy()

    # 6. Process Ground Truth Mask
    # Squeeze dimensions in case the mask is stored as [1, H, W]
    ground_truth = sample_dict["mask"][sample_idx].cpu().numpy().squeeze()

    # 7. Extract Labels, Scores, and Remarks
    true_label = sample_dict["label"][sample_idx].item()
    pred_label = sample_dict["pred_labels"][sample_idx].item()
    classification_rem = classification_remark(true_label, pred_label)

    pred_score = sample_dict["pred_scores"][sample_idx].item()
    pred_thresh = sample_dict.get("pred_threshold", "N/A")

    # Retrieve segment/count info (fallback to pred_labels if 'segments' key is missing)
    segments_info = sample_dict.get("segments", sample_dict["pred_labels"])
    total_preds = segments_info[sample_idx] if isinstance(segments_info, list) else segments_info[sample_idx].item()

    # 8. Configure Main Figure Title
    fig.suptitle(f"True Label: {true_label}, Predicted Label: {pred_label}\n"
                 f"Score: {pred_score:.4f} | Threshold: {pred_thresh}\n"
                 f"Remark: {classification_rem}\n"
                 f"Path: {os.path.basename(actual_path)}", fontsize=10)

    # 9. Plotting Subplots
    ax[0, 0].imshow(img_resized)
    ax[0, 0].set_title("Original (Resized)")

    ax[0, 1].imshow(ground_truth, cmap='gray')
    ax[0, 1].set_title("Ground Truth")

    ax[1, 0].imshow(heat_map)
    ax[1, 0].set_title("Anomaly Heatmap")

    ax[1, 1].imshow(pred_mask, cmap='gray', interpolation='nearest')
    ax[1, 1].set_title(f"Predicted Segments: {total_preds}")

    # Hide axes for a cleaner visual layout
    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()
    return plt


def save_all_predictions(
        predictions: List[Dict],
        output_dir: str,
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Initialize index counter
    global_index = 0

    min_pred_value = float('inf')
    max_pred_value = -float('inf')
    for batch in predictions:
        batch_anomaly_maps = batch["anomaly_maps"]  # Shape: [B, C, H, W]
        batch_min = batch_anomaly_maps.min().item()
        batch_max = batch_anomaly_maps.max().item()
        if batch_min < min_pred_value:
            min_pred_value = batch_min
        if batch_max > max_pred_value:
            max_pred_value = batch_max

    # Loop through all batches in predictions
    for batch_idx, batch in enumerate(predictions):
        batch_size = len(batch['label'])

        # Loop through all samples in current batch
        for sample_idx in range(batch_size):
            try:
                # 1. Generate the plot using the function we updated
                plot = visualiser(batch, sample_idx, min_pred_value, max_pred_value)

                # 2. Safely extract the image path string
                raw_p = batch["image_path"][sample_idx]
                image_path = raw_p[0] if isinstance(raw_p, list) and not isinstance(raw_p, str) else raw_p

                # 3. Get the filename without extension
                # os.path.splitext returns (root, ext), so we take [0]
                file_name_only = os.path.splitext(os.path.basename(image_path))[0]

                # 4. Construct the save path
                save_path = os.path.join(output_dir, f"{file_name_only}_prediction.png")

                # 5. Save and Cleanup
                plot.savefig(save_path, bbox_inches='tight', dpi=300)
                plot.close()

                global_index += 1

            except Exception as e:
                print(f"Error processing batch {batch_idx}, sample {sample_idx}: {str(e)}")
                continue


def classification_remark(label: int, prediction: bool):
    if label == 1:
        if prediction == True:
            return "tp"
        else:
            return "fn"
    if label == 0:
        if prediction == True:
            return "fp"
        else:
            return "tn"
    else:
        raise ValueError("Error label not 0 or 1")


def visualise_metrics(predictions: List[Dict], save_path: str = None):
    metric1 = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for batchidx, batch in enumerate(predictions):
        batch_size = len(batch['label'])
        for sample_idx in range(batch_size):
            true_label = batch["label"][sample_idx]
            pred_label = batch["pred_labels"][sample_idx]
            metric1[classification_remark(true_label, pred_label)] += 1
    TP = metric1['tp']
    FN = metric1['fn']
    FP = metric1['fp']
    TN = metric1['tn']

    # Calculating Metrics
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # Also TPR (Sensitivity)
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    metrics = metric1 | {
        "Precision": precision,
        "Recall (TPR)": recall,
        "FPR": fpr,
        "F1 Score": f1_score,
        "Accuracy": accuracy,

    }

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Plotting the table
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    table_data = [[name, f"{value:.4f}"] for name, value in zip(metric_names, metric_values)]
    table = ax.table(cellText=table_data, colLabels=["Image Metric", "Value"], cellLoc="center", loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Show plot
    plt.show()


def image_auroc(test_dat,save=""):
    """
    Calculate AUROC for each image in the test dataset.
    Args:
        test_dat (list): List of dictionaries containing test data.
    Returns:
        list: List of AUROC scores for each image.
    """
    image_auroc_metric = AUROC(thresholds=400)
    true_labels = torch.cat([x['label'] for x in test_dat])
    pred_scores = torch.cat([x['pred_scores'] for x in test_dat])
    image_score = image_auroc_metric(pred_scores, true_labels)
    print("Image AUROC SCORE:", image_score)

    fpr, tpr = image_auroc_metric._compute()
    optimal_threshold_idx = np.argmax(tpr - fpr)
    optimal_thr = image_auroc_metric.thresholds[optimal_threshold_idx]
    print("Optimal Threhsold:", optimal_thr)

    if save!="":
        fig, title = image_auroc_metric.generate_figure()
        fig.savefig('img_level_auroc.png')


def pixel_auroc(test_dat,save=""):
    """
    Calculate pixel AUROC for each image in the test dataset.
    Args:
        test_dat (list): List of dictionaries containing test data.
    Returns:
        list: List of AUROC scores for each image.
    """
    pixel_auroc_metric = AUROC(thresholds=300)
    true_masks = torch.cat([x['mask'].flatten() for x in test_dat])
    pred_anomaly_maps = torch.cat([x['anomaly_maps'].flatten() for x in test_dat])
    pixel_score = pixel_auroc_metric(pred_anomaly_maps, true_masks)
    print("Pixel AUROC ", pixel_score)
    if save!="":
        fig, title = pixel_auroc_metric.generate_figure()
        fig.savefig('pixel_level_auroc.png')


def pro_metric(test_data,save=""):
    """
    Calculate PRO metric for each image in the test dataset.
    Args:
        preds (list): List of dictionaries containing test data.
    Returns:
        list: List of PRO scores for each image.
    """
    pro_evaluator= PRO()
    true_masks = torch.cat([x['mask'] for x in test_data])
    pred_anomaly_maps = torch.cat([x['anomaly_maps'] for x in test_data])
    for gt,pred in zip(true_masks,pred_anomaly_maps):
        pro_evaluator.update(pred, gt)
    pro_score= pro_evaluator.compute()
    print("PRO score:", pro_score)
    if save!="":
        fig, title = pro_evaluator.generate_figure()
        fig.savefig('pro_level.png')


def parse_args():
    parser = argparse.ArgumentParser(description="Train Patchcore anomaly detection model")
    parser.add_argument(
        "model_output_dir", type=str,
        help="Directory where the trained model will be saved"
    )
    parser.add_argument(
        "dataset_dir", type=str,
        help="Path to dataset directory (raw or containing the tiled one)"
    )
    parser.add_argument(
        "--is_dataset_splitted", action="store_true",
        help="Flag indicating the dataset is already tiled"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="model_split_per_partnorm",
        help="Name of the experiment"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare dataset directory
    # dataset_patches_dir = os.path.join(args.dataset_dir, "dataset_patches_new3")
    dataset_patches_dir = os.path.join(args.dataset_dir, "dataset_cropped_candy")
    # Define dataset directories
    normal_dir_op = os.path.join(dataset_patches_dir, "train")
    normal_test_dir_op = os.path.join(dataset_patches_dir, "test", "normal")
    abnormal_dir_op = os.path.join(dataset_patches_dir, "test", "anomaly")
    mask_dir_op = os.path.join(dataset_patches_dir, "masks_candy")

    dataset_dirs_op = {
        "normal_dir": normal_dir_op,
        "normal_test_dir": normal_test_dir_op,
        "abnormal_dir": abnormal_dir_op,
        "mask_dir": mask_dir_op,
    }

    if not args.is_dataset_splitted:     #如果图片没有被裁剪
        print("Tiling the dataset...")
        create_tiled_dataset(args.dataset_dir, dataset_dirs_op)

    # Set model save path and batch size
    model_pth = os.path.join(args.model_output_dir, args.experiment_name)
    # Ensure output directory exists
    os.makedirs(model_pth, exist_ok=True)
    batch_sz = 4  # don't change



    # Compute per-patch normalization values
    # dirs = ["left_patches", "middle_patches", "right_patches"]
    dirs = [""]

    lis = {}

    preds = []
    thres = []

    for base_dir in dirs:

        mydataset = Folder(
            name="RESIZE_NORMALIZED",
            task=TaskType.SEGMENTATION,
            normal_dir=os.path.join(dataset_dirs_op["normal_dir"], base_dir),
            normal_test_dir=os.path.join(dataset_dirs_op["normal_test_dir"], base_dir),
            abnormal_dir=os.path.join(dataset_dirs_op["abnormal_dir"], base_dir),
            mask_dir=os.path.join(dataset_dirs_op["mask_dir"], base_dir),
            val_split_mode='same_as_test',
            test_split_mode='from_dir',
            test_split_ratio=0,
            val_split_ratio=0,
            normal_split_ratio=0,
            num_workers=8,
            train_batch_size=batch_sz,
            eval_batch_size=batch_sz,
            # train_transform=train_transform,
            # eval_transform=eval_transform,
            seed=42
        )
        mydataset.prepare_data()
        mydataset.setup()

        # To find normalizations values with  images
        sum_rgb = torch.zeros(3)
        squared_sum_rgb = torch.zeros(3)
        num_pixels = 0

        # Iterate over the dataset
        for img in mydataset.train_data:
            imgg = img["image"]
            sum_rgb += imgg.sum(dim=[1, 2])
            squared_sum_rgb += (imgg ** 2).sum(dim=[1, 2])
            num_pixels += imgg.numel() / 3

        # Calculate mean and std
        mean = sum_rgb / num_pixels
        std = (squared_sum_rgb / num_pixels - mean ** 2) ** 0.5

        lis[base_dir] = {'mean': mean, 'std': std}

        train_transform = Compose(
            [

                # Resize((720,1280)),
                # Resize((256, 256)),
                # CenterCrop((300,1240)),
                Normalize(mean=mean, std=std),  # dataset normalization
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ],
        )
        eval_transform = Compose(
            [
                # Resize((720,1280)),
                # Resize((256, 256)),
                # CenterCrop((300,1240)),
                Normalize(mean=mean, std=std),  # dataset normalization
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ],
        )

        mydataset_for_train = Folder(
            name="RESIZE_NORMALIZED",
            task=TaskType.SEGMENTATION,
            normal_dir=os.path.join(dataset_dirs_op["normal_dir"], base_dir),
            normal_test_dir=os.path.join(dataset_dirs_op["normal_test_dir"], base_dir),
            abnormal_dir=os.path.join(dataset_dirs_op["abnormal_dir"], base_dir),
            mask_dir=os.path.join(dataset_dirs_op["mask_dir"], base_dir),
            val_split_mode='same_as_test',
            test_split_mode='from_dir',
            test_split_ratio=0,
            val_split_ratio=0,
            normal_split_ratio=0,
            num_workers=8,
            train_batch_size=batch_sz,
            eval_batch_size=batch_sz,
            train_transform=train_transform,
            eval_transform=eval_transform,
            seed=42
        )

        mydataset_for_train.prepare_data()
        mydataset_for_train.setup()

        # define model and the metrics
        model = Patchcore()
        engine = Engine(
            image_metrics=['AUROC'],
            pixel_metrics=['AUROC', 'PRO'],
            default_root_dir=os.path.join(model_pth, base_dir))

        # Fit the model
        engine.fit(model=model, datamodule=mydataset_for_train)

        # test model
        # mytests = engine.test(model=model, datamodule=mydataset_for_train, verbose=True)

        # Load the model and get predictions
        model = Patchcore.load_from_checkpoint(os.path.join(model_pth, base_dir,"Patchcore/RESIZE_NORMALIZED/latest/weights/lightning/model.ckpt"))
        engine = Engine()
        preds_ = engine.predict(model=model, dataloaders=mydataset_for_train.test_dataloader())
        img_thresh = model.image_threshold.value.item()
        pxl_thresh = model.pixel_threshold.value.item()

        preds.append(preds_)
        thres.append((img_thresh, pxl_thresh))

    preds_combined=[]
    for pred,thrs in zip(preds,thres):
        tmp=[]
        for batches in pred:
            new_batch = {}
            new_batch['image'] = batches['image']
            new_batch['image_path'] = batches['image_path']
            new_batch['label'] = batches['label']
            new_batch['pred_threshold'] = thrs[0]
            new_batch['anomaly_threshold'] = thrs[1]
            new_batch['pred_nonormalized'] = batches['pred_scores']
            new_batch['anomaly_nonormalized_max'] = batches['anomaly_maps'].max()
            new_batch['pred_labels'] = batches['pred_labels']
            new_batch['pred_masks'] =batches['pred_masks']
            new_batch['pred_scores'] = batches['pred_scores']
            m = batches['mask']
            new_batch['mask'] = m.unsqueeze(1) if m.ndim == 3 else m
            new_batch['anomaly_maps'] = batches['anomaly_maps']
            # new_batch['segments'] = [len(t.tolist()) for t in batches["box_labels"]]
            # Check if 'box_labels' key exists; if not, calculate from predicted masks
            if "box_labels" in batches:
                # Use the length of existing box labels (Ground Truth or model-generated)
                new_batch['segments'] = [len(t.tolist()) for t in batches["box_labels"]]
            else:
                # Fallback: Count disconnected anomalous regions using Connected Components Analysis (CCA)
                seg_counts = []
                for pred_m in batches["pred_masks"]:
                    # Convert tensor to NumPy uint8 format
                    # Scale 0/1 to 0/255 and remove redundant dimensions (e.g., [1, H, W] -> [H, W])
                    mask_np = (pred_m.cpu().numpy() * 255).astype(np.uint8).squeeze()

                    # Perform Connected Components Analysis
                    # num_labels includes the background, so subtract 1 to get the actual defect count
                    num_labels, _ = cv2.connectedComponents(mask_np)

                    # Ensure count is non-negative and append to list
                    seg_counts.append(max(0, num_labels - 1))

                new_batch['segments'] = seg_counts

            tmp.append(new_batch)
        preds_combined.append(tmp)

    # test_dat=[]
    # for left,middle,right in zip(preds_combined[0]):
    #     data ={}
    #     data['image_path'] =[ [left_img,middle_img,right_img ] for left_img,middle_img,right_img in zip(left['image_path'],middle['image_path'],right['image_path']) ]
    #     labels = torch.stack([left['label'],middle['label'],right['label']])
    #     data['label'] = labels.max(dim=0).values
    #     pred_scores_stacked = torch.stack([left['pred_scores'],middle['pred_scores'],right['pred_scores']])
    #     data['pred_scores'],_= torch.max(pred_scores_stacked, dim=0)
    #     pred_labels_stacked = torch.stack([left['pred_labels'],middle['pred_labels'],right['pred_labels']])
    #     data['pred_labels'],_= torch.max(pred_labels_stacked, dim=0)
    #     data['pred_masks'] = torch.cat((left['pred_masks'],middle['pred_masks'],right['pred_masks']),dim=3)
    #     data['mask'] = torch.cat((left['mask'],middle['mask'],right['mask']),dim=3).squeeze(1)
    #     data['anomaly_maps'] = torch.cat((left['anomaly_maps'],middle['anomaly_maps'],right['anomaly_maps']),dim=3)
    #     data['segments'] = [left['segments'][i] + middle['segments'][i] + right['segments'][i] for i in range(len(left['segments']))]
    #     data['pred_threshold'] = [left['pred_threshold'],middle['pred_threshold'],right['pred_threshold']]
    #     data['anomaly_threshold'] = [left['anomaly_threshold'],middle['anomaly_threshold'],right['anomaly_threshold']]
    #     data['pred_nonormalized'] = torch.max(torch.stack([left['pred_nonormalized'],middle['pred_nonormalized'],right['pred_nonormalized']]),
    #                                           dim=0).values.tolist()
    #     data['anomaly_nonormalized'] =torch.max( torch.stack([left['anomaly_nonormalized_max'],middle['anomaly_nonormalized_max'],right['anomaly_nonormalized_max']]),
    #                                             dim=0).values.tolist()
    #     test_dat.append(data)

    test_dat = preds_combined[0]
    # Specify a path if you want to save the evaluated curves
    image_auroc(test_dat,"yes"), pixel_auroc(test_dat,"yes"), pro_metric(test_dat,"yes")
    visualise_metrics(test_dat)
    save_all_predictions(test_dat, "split_per_candy_output")


if __name__ == "__main__":
    main()