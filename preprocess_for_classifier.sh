#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --scores_csv <scores_csv_path> --bbox_file <bbox_file_path> --image_dir <image_dir_path> --split_subsets_by_id <split_subsets_by_id>"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scores_csv)
            SCORES_CSV="$2"
            shift 2
            ;;
        --bbox_file)
            BBOX_FILE="$2"
            shift 2
            ;;
        --image_dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --split_subsets_by_id)
            SPLIT_SUBSETS_BY_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Check if all required arguments are provided
if [[ -z "$SCORES_CSV" || -z "$BBOX_FILE" || -z "$IMAGE_DIR" || -z "$SPLIT_SUBSETS_BY_ID" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Get the full path of the current directory
CURRENT_DIR=$(pwd)

# Add the current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR
echo "Updated PYTHONPATH: $PYTHONPATH"

# Step 1: Run average_scores.py
echo "Running data_utils/average_scores.py..."
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: data/averaged_scores.csv"
python data_utils/average_scores.py --input_csv "$SCORES_CSV" --output_csv data/averaged_scores.csv
if [ $? -eq 0 ]; then
    echo "Averaged scores saved to data/averaged_scores.csv"
else
    echo "Error: Failed to run average_scores.py"
    exit 1
fi

# Step 2: Run merge_scores_bbox_files.py
echo "Running data_utils/merge_scores_bbox_files.py..."
echo "Bounding Box File: $BBOX_FILE"
echo "Score File: data/averaged_scores.csv"
echo "Output File: data/merged_score_file.csv"
python data_utils/merge_scores_bbox_files.py --bbox_file "$BBOX_FILE" --score_file data/averaged_scores.csv --output_file data/merged_score_file.csv
if [ $? -eq 0 ]; then
    echo "Merged scores and bounding boxes saved to data/merged_score_file.csv"
else
    echo "Error: Failed to run merge_scores_bbox_files.py"
    exit 1
fi

# Step 3: Run crop_images.py
echo "Running data_utils/crop_images.py..."
echo "Label File: data/merged_score_file.csv"
echo "Image Directory: $IMAGE_DIR"
echo "Output Directory: data/classifier_data"
echo "Split Subsets By ID: $SPLIT_SUBSETS_BY_ID"
python data_utils/crop_images.py --label_file data/merged_score_file.csv --image_dir "$IMAGE_DIR" --output_dir data/classifier_data --split_subsets_by_id "$SPLIT_SUBSETS_BY_ID"
if [ $? -eq 0 ]; then
    echo "Cropped images saved to data/classifier_data"
else
    echo "Error: Failed to run crop_images.py"
    exit 1
fi

echo "All steps completed successfully!"