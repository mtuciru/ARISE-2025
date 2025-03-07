import argparse
import cv2
import pandas as pd
import os
import json
from PIL import Image


def main(data, img_dir, output_dir, subsets):
    """
    Preprocess data for YOLO format.

    Args:
        data (pd.DataFrame): Input data with columns: patient_id, joint, xcenter, ycenter, dx, dy.
        img_dir (str) : Directory containing images for YOLO format.
        output_dir (str): Directory to save the YOLO annotation files.
    """
    # Create output directory if it doesn't exist
    images_train_dir = os.path.join(output_dir, "images/train")
    images_val_dir = os.path.join(output_dir, "images/val")
    images_test_dir = os.path.join(output_dir, "images/test")

    labels_train_dir = os.path.join(output_dir, "labels/train")
    labels_val_dir = os.path.join(output_dir, "labels/val")
    labels_test_dir = os.path.join(output_dir, "labels/test")


    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    os.makedirs(labels_test_dir, exist_ok=True)

    # Group data by patient_id (each patient_id corresponds to one photo)
    grouped = data.groupby('patient_id')

    for patient_id, group in grouped:
        img_path = os.path.join(img_dir, f"{patient_id}" + ".jpeg")
        img = Image.open(img_path)
        w, h = img.size

        if patient_id in subsets["train"]:
            save_img = images_train_dir
            save_label = labels_train_dir
        elif patient_id in subsets["val"]:
            save_img = images_val_dir
            save_label = labels_val_dir
        else:
            save_img = images_test_dir
            save_label = labels_test_dir


        img.save(os.path.join(save_img, f"{patient_id}.jpeg"))

        txt_file = os.path.join(save_label, f"{patient_id}.txt")

        f = open(txt_file, "w")

        for _, row in group.iterrows():
            # Extract bounding box details
            x_center = row['xcenter']
            y_center = row['ycenter']
            dx = row['dx']
            dy = row['dy']

            # Normalize coordinates for YOLO
            x_center_norm = x_center / (w)
            y_center_norm = y_center / h
            width_norm = dx / (w)
            height_norm = dy / h


            # Write to the .txt file in YOLO format
            f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        f.close()
        print(f"Created YOLO annotation file: {txt_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for YOLO format.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save YOLO annotation files.")
    parser.add_argument("--split_info_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.split_info_path, "r") as f:
        subsets = json.load(f)
    # Load the data
    data = pd.read_csv(args.csv)

    # Run preprocessing
    main(data, args.img_dir, args.output_dir, subsets)
