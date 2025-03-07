import os
import cv2
import pandas as pd
from PIL import Image

def crop_and_save_images(image_dir, bbox_txt_path, output_dir):
    """
    Crop images based on bounding boxes from a text file and save them in the specified format.

    Args:
        image_dir (str): Path to the directory containing input images.
        bbox_txt_path (str): Path to the bounding box text file.
        output_dir (str): Path to save the cropped images.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the bounding box text file into a DataFrame
    bbox_df = pd.read_csv(bbox_txt_path)

    # Dictionary to keep track of how many times each patient_id has been seen
    patient_id_count = {}

    # Iterate over each row in the bounding box DataFrame
    for _, row in bbox_df.iterrows():
        patient_id = str(int(row["patient_id"]))
        xcenter = row["xcenter"]
        ycenter = row["ycenter"]
        dx = row["dx"]
        dy = row["dy"]

        # Load the image
        image_path = os.path.join(image_dir, f"{patient_id}.jpeg")
        if not os.path.exists(image_path):
            print(f"Image not found for patient_id {patient_id}: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Get image dimensions
        height, width, _ = image.shape

        # Convert normalized coordinates to pixel coordinates
        x_center_px = int(xcenter * width)
        y_center_px = int(ycenter * height)
        dx_px = int(dx * width)
        dy_px = int(dy * height)

        # Calculate bounding box coordinates
        x1 = x_center_px - dx_px // 2
        y1 = y_center_px - dy_px // 2
        x2 = x_center_px + dx_px // 2
        y2 = y_center_px + dy_px // 2

        # Ensure the bounding box is within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # Update the count for this patient_id

        # Save the cropped image
        output_image_name = f"{patient_id}_{int(row['joint_id'])}.jpeg"
        output_image_path = os.path.join(output_dir, output_image_name)
        cv2.imwrite(output_image_path, cropped_image)
        print(f"Saved cropped image: {output_image_path}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Crop images based on bounding boxes from a text file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument("--bbox_txt", type=str, required=True, help="Path to the bounding box text file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the cropped images.")
    args = parser.parse_args()

    # Run the cropping function
    crop_and_save_images(args.image_dir, args.bbox_txt, args.output_dir)