usage() {
    echo "Usage: $0 --csv <csv_path> --img_dir <img_dir_path> --split_info_path <split_info_path>"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)
            CSV_PATH="$2"
            shift 2
            ;;
        --img_dir)
            IMG_DIR="$2"
            shift 2
            ;;
        --split_info_path)
            SPLIT_INFO_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Check if all required arguments are provided
if [[ -z "$CSV_PATH" || -z "$IMG_DIR" || -z "$SPLIT_INFO_PATH" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Get the full path of the current directory
CURRENT_DIR=$(pwd)

# Add the current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR
echo "Updated PYTHONPATH: $PYTHONPATH"

# Run the Python script with the provided arguments
python data_utils/process_for_yolo.py \
    --csv "$CSV_PATH" \
    --img_dir "$IMG_DIR" \
    --output_dir "data/yolo_dataset" \
    --split_info_path "$SPLIT_INFO_PATH"
