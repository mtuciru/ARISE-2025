# ARISE-2025: Automated Rheumatoid Inflammatory Scoring and Evaluation

## Task Description

Rheumatoid arthritis (RA) is a chronic autoimmune disease characterized by inflammation, joint destruction, and extra-articular manifestations. Radiography is the standard imaging modality for diagnosing and monitoring joint damage in RA. However, traditional methods for evaluating radiographic progression, such as the Sharp method and its variants, are time-consuming and subjective. This hackathon focuses on developing automated solutions for joint assessment in RA using computer vision techniques.

Participants will build models to automatically score hand joints affected by RA. The task involves two key components:
1. **Joint Localization**: Accurately localize hand joints in radiographic images.
2. **Pathology Assessment**: Evaluate the severity of joint damage, specifically focusing on **erosion** and **joint space narrowing (JSN)** and predict damage scores (0-4 for JSN and 0-5 for erosion).

The goal is to develop a robust and efficient pipeline that can assist clinicians in diagnosing and monitoring RA progression, reducing subjectivity and manual effort.

---

## Evaluation Metrics

The performance of the models will be evaluated using the following metrics:

### 1. **Intersection over Union (IoU)**
IoU measures the overlap between the predicted bounding box and the ground truth bounding box. It is defined as:

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

- **Area of Overlap**: The region where the predicted and ground truth bounding boxes intersect.
- **Area of Union**: The total area covered by both the predicted and ground truth bounding boxes.

A higher IoU indicates better localization accuracy.

---

### 2. **Balanced Accuracy**
Balanced Accuracy is a metric used to evaluate the performance of a classification model, especially in cases where the classes are imbalanced. It is the average of recall (sensitivity) obtained on each class, ensuring that the performance metric is not biased toward the majority class.

Balanced Accuracy is defined as:

$$
\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}} + \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}} \right)
$$

---

### 3. **Final Metric: IoU × Accuracy**
The final evaluation metric is the product of **IoU** and **Accuracy**. This combined metric ensures that models achieve both precise localization and accurate pathology assessment. It is defined as:

$$
\text{Final Metric} = \text{IoU} \times \text{Accuracy}
$$

The final score ranges between 0 and 1, where higher values indicate better overall performance.

---

## Dataset

Participants will work with a clinical collection of radiographic images annotated for joint localization, erosion, and JSN. The dataset includes:
- **Images**: High-resolution radiographic images of hand joints.
- **Annotations**: Bounding boxes for joint localization and severity scores for erosion and JSN.

---

## Expected Outcomes

Participants are expected to develop a pipeline that:
1. Accurately localizes hand joints in radiographic images.
2. Assesses the severity of erosion and JSN with high precision.
3. Demonstrates generalizability and robustness across diverse patient data.

The winning solutions will be those that achieve the highest performance on the evaluation metrics while providing interpretable and clinically relevant results.

---

## Important Notes for Preprocessing and Submission

### 1. **Handling Multiple Expert Scores**
- For each joint in the `scores.csv` file, there are three entries with scores from three different experts. It is recommended to use the **average value** as the ground truth. This averaging is already implemented in the repository's preprocessing code. If you use a different preprocessing approach, ensure that you average the scores accordingly.

### 2. **Submission File Format (`submit.csv`)**
- Each image must have exactly **100 rows** in the `submit.csv` file. If there are fewer than 100 joints for a patient, pad the remaining rows with empty values.
- Padding rows should have the following format:
  - `PAD` column set to `1`.
  - All other columns (`joint_id`, `xcenter`, `ycenter`, `dx`, `dy`, `jsn_score`, `erosion_score`) set to `none` (str 'none', not empty value!).

#### Example of Padding Rows:
| ID      | patient_id | joint_id | xcenter | ycenter | dx   | dy   | jsn_score | erosion_score | PAD |
|---------|------------|----------|---------|---------|------|------|-----------|---------------|-----|
| 6_48    | 6          | none     | none    | none    | none | none | none      | none          | 1.0 |

#### Example of Predicted Rows:
| ID      | patient_id | joint_id | xcenter            | ycenter            | dx                | dy                | jsn_score | erosion_score | PAD |
|---------|------------|----------|--------------------|--------------------|-------------------|-------------------|-----------|---------------|-----|
| 1_14    | 6          | 14       | 0.8670480847358704 | 0.2998411357402801 | 0.0537742339074611 | 0.0342460758984088 | 2         | 0             | 0.0 |

- The `ID` column should follow the format: `{patient_id}_{joint_index}` (e.g., `1_14` for the 14th joint of patient 1.
- __You are not required to follow a specific numbering scheme for joint IDs__. However, each patient must have exactly 100 unique IDs in the submission file to meet Kaggle's processing requirements. Matching joints to ground truth will be handled automatically using Intersection over Union (IoU).
- __Bounding box coordinates (xcenter, ycenter, dx, dy) must be normalized to the image dimensions, meaning all values must be between 0 and 1__.
---

# Baseline Solution

### Troubleshooting Import Issues

If you encounter import problems, ensure that the project directory is added to your `PYTHONPATH`. Run the following command in your terminal:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/arise-2025
```
Replace /path/to/arise-2025 with the absolute path to the root directory of the project. This ensures that Python can locate and import the necessary modules.


## Training

As a baseline, we present a pipeline combining **YOLOv12s** for joint localization and **ResNet50** for pathology assessment.

### Preprocessing for Detection Model

```bash
chmod +x preprocess_for_yolo.sh
./preprocess_for_yolo.sh --csv /path/to/csv --img_dir /path/to/images --split_info_path /path/to/split_info.json
```

### Train YOLO

1. Add your `yolo_dataset` path to the detection config.
2. Run the training script:

```bash
python detection/model/yolo/train.py
```

## Preprocessing Data for Classifier Training

The classifier (ResNet50) is trained on ground-truth bounding boxes. The preprocessing steps include:
- Averaging expert scores.
- Merging scores and bounding box files.
- Normalizing bounding box coordinates and cropping images.

```bash
chmod +x preprocess_for_classifier.sh
./preprocess_for_classifier.sh --scores_csv /path/to/scores.csv --bbox_file /path/to/bboxes.csv --image_dir /path/to/jpeg --split_subsets_by_id /path/to/train_val_split.json
```

### Training the Classifier

1. Add your Weights & Biases (wandb) key to `config/classifier/train.yaml`.
2. Configure the model in the config file (any model can be initialized via Hydra).
3. Run the training script:

```bash
python classifier/train.py
```

---

## Evaluate and Submit

### Detection

1. Detect joints on evaluation data:

```bash
python detection/model/yolo/infer.py --model /path/to/model_weights --img_dir /path/to/eval_data --output_dir /path/to/output_dir --imgsz pretrained_model_image_size
```

2. Crop images using the generated bounding boxes:

```bash
python data_utils/crop_eval.py --image_dir /path/to/image/jpeg --bbox_txt /path/to/detection/all_bboxes.txt --output_dir data/eval_cropped_images
```

### Classification

1. Predict labels for cropped images (edit `config/classification/submit.yaml`):
   - `model_weights`: Path to the best classification model checkpoint.
   - `bbox_csv`: Path to predicted bounding boxes from the detection model.
   - `image_dir`: Path to cropped images (`data/eval_cropped_images`).
   - `output_csv`: Path to save the submission file (`submit.csv`).

2. Run the submission script:

```bash
python submit.py
```

---

## Submission File Requirements

- Ensure the `submit.csv` file adheres to the specified format.
- Normalize bounding box coordinates (`xcenter`, `ycenter`, `dx`, `dy`) to the image dimensions, meaning all values must be between **0 and 1**.
- Include exactly 100 rows per image, padding with empty rows if necessary.
- Use the `PAD` column to indicate padding rows (`1` for padding, `0` for predicted rows).
- **ID System**:
  - Each patient must have exactly 100 unique IDs in the format `{patient_id}_{joint_index}` (e.g., `1_14` for the 14th joint of patient 1).
  - You are not required to follow a specific numbering scheme for joint IDs, but the IDs must be unique for each patient.
  - Matching joints to ground truth will be handled automatically using Intersection over Union (IoU).
