import os
import argparse
from ultralytics import YOLO

def inference_on_directory(model_path, image_dir, output_dir, imgsz):
    """
    Perform inference on all images in a directory using a YOLO model.

    Args:
        model_path (str): Path to the trained YOLOv model (.pt file).
        image_dir (str): Path to the directory containing input images.
        output_dir (str): Path to save the output images with predictions.
        imgsz: Image size.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Create output directories
    output_images_dir = os.path.join(output_dir, "images_with_bboxes")
    output_txts_dir = os.path.join(output_dir, "bbox_txts")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_txts_dir, exist_ok=True)

    # Create a single output text file for all bounding boxes
    output_txt_path = os.path.join(output_txts_dir, "all_bboxes.txt")
    with open(output_txt_path, "w") as f:
        # Write the header
        f.write("patient_id,joint_id,xcenter,ycenter,dx,dy\n")

        # Dictionary to keep track of joint_id for each patient_id
        patient_id_counter = {}

        # Iterate over all images in the input directory
        for image_name in os.listdir(image_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(image_dir, image_name)
                print(f"Processing image: {image_path}")

                # Extract patient_id from the image name (assuming format: {patient_id}.jpg)
                patient_id = int(image_name.split(".")[0])

                # Perform inference
                results = model.predict(
                    source=image_path,
                    imgsz=imgsz,
                    save=False,  # Disable automatic saving
                )

                # Get the first result (assuming single image inference)
                result = results[0]

                # Save the image with bounding boxes (without class names)
                output_image_path = os.path.join(output_images_dir, image_name)
                result.save(filename=output_image_path)

                # Save bounding box coordinates to the single .txt file
                for box in result.boxes:
                    # Get bounding box coordinates (x_center, y_center, width, height)
                    x_center, y_center, width, height = box.xywhn[0].tolist()

                    # Update joint_id for this patient_id
                    if patient_id not in patient_id_counter:
                        patient_id_counter[patient_id] = 0
                    patient_id_counter[patient_id] += 1
                    joint_id = patient_id_counter[patient_id]

                    # Write to file (patient_id,joint_id,xcenter,ycenter,dx,dy)
                    f.write(f"{patient_id},{joint_id},{x_center},{y_center},{width},{height}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on images using a YOLO model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained YOLO model (.pt file).")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output images and bounding box files.")
    parser.add_argument("--imgsz", type=int, required=True, help="Image size.")
    args = parser.parse_args()

    inference_on_directory(args.model, args.img_dir, args.output_dir, args.imgsz)
