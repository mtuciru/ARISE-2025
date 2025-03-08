# ARISE-2025: Automated Rheumatoid Inflammatory Scoring and Evaluation

## Task Description

Rheumatoid arthritis (RA) is a chronic autoimmune disease characterized by inflammation, joint destruction, and extra-articular manifestations. Radiography is the standard imaging modality for diagnosing and monitoring joint damage in RA. However, traditional methods for evaluating radiographic progression, such as the Sharp method and its variants, are time-consuming and subjective. This hackathon focuses on developing automated solutions for joint assessment in RA using computer vision techniques.

Participants will build models to automatically score hand joints affected by RA. The task involves two key components:
1. \textbf{Joint Localization}: Accurately localize hand joints in radiographic images.
2. \textbf{Pathology Assessment}: Evaluate the severity of joint damage, specifically focusing on \textbf{erosion} and \textbf{joint space narrowing (JSN)} and predict damage scores (0-4 for JSN and 0-5 for erosion).

The goal is to develop a robust and efficient pipeline that can assist clinicians in diagnosing and monitoring RA progression, reducing subjectivity and manual effort.

---

## Evaluation Metrics

The performance of the models will be evaluated using the following metrics:

### 1. \textbf{Intersection over Union (IoU)}
IoU measures the overlap between the predicted bounding box and the ground truth bounding box. It is defined as:

\[
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
\]

- \textbf{Area of Overlap}: The region where the predicted and ground truth bounding boxes intersect.
- \textbf{Area of Union}: The total area covered by both the predicted and ground truth bounding boxes.

A higher IoU indicates better localization accuracy.

---

### 2. \textbf{Balanced Accuracy}
Balanced Accuracy is a metric used to evaluate the performance of a classification model, especially in cases where the classes are imbalanced. It is the average of recall (sensitivity) obtained on each class, ensuring that the performance metric is not biased toward the majority class.

Balanced Accuracy is defined as:

\[
\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}} + \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}} \right)
\]

---

### 3. \textbf{Final Metric: IoU × Accuracy}
The final evaluation metric is the product of \textbf{IoU} and \textbf{Accuracy}. This combined metric ensures that models achieve both precise localization and accurate pathology assessment. It is defined as:

\[
\text{Final Metric} = \text{IoU} \times \text{Accuracy}
\]

The final score ranges between 0 and 1, where higher values indicate better overall performance.

---

## Dataset

Participants will work with a clinical collection of radiographic images annotated for joint localization, erosion, and JSN. The dataset includes:
- \textbf{Images}: High-resolution radiographic images of hand joints.
- \textbf{Annotations}: Bounding boxes for joint localization and severity scores for erosion and JSN.

---

## Expected Outcomes

Participants are expected to develop a pipeline that:
1. Accurately localizes hand joints in radiographic images.
2. Assesses the severity of erosion and JSN with high precision.
3. Demonstrates generalizability and robustness across diverse patient data.

The winning solutions will be those that achieve the highest performance on the evaluation metrics while providing interpretable and clinically relevant results.

---

## Important Notes for Preprocessing and Submission

### 1. \textbf{Handling Multiple Expert Scores}
- For each joint in the \texttt{scores.csv} file, there are three entries with scores from three different experts. It is recommended to use the \textbf{average value} as the ground truth. This averaging is already implemented in the repository's preprocessing code. If you use a different preprocessing approach, ensure that you average the scores accordingly.

### 2. \textbf{Submission File Format (\texttt{submit.csv})}
- Each image must have exactly \textbf{100 rows} in the \texttt{submit.csv} file. If there are fewer than 100 joints for a patient, pad the remaining rows with empty values.
- Padding rows should have the following format:
  - \texttt{PAD} column set to \texttt{1}.
  - All other columns (\texttt{joint\_id}, \texttt{xcenter}, \texttt{ycenter}, \texttt{dx}, \texttt{dy}, \texttt{jsn\_score}, \texttt{erosion\_score}) set to \texttt{none}.

#### Example of Padding Rows:
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
\hline
ID & patient\_id & joint\_id & xcenter & ycenter & dx & dy & jsn\_score & erosion\_score & PAD \\
\hline
6\_48 & 6 & none & none & none & none & none & none & none & 1.0 \\
\hline
\end{tabular}

#### Example of Predicted Rows:
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
\hline
ID & patient\_id & joint\_id & xcenter & ycenter & dx & dy & jsn\_score & erosion\_score & PAD \\
\hline
1\_14 & 6 & 14 & 0.8670480847358704 & 0.2998411357402801 & 0.0537742339074611 & 0.0342460758984088 & 2 & 0 & 0.0 \\
\hline
\end{tabular}

- The \texttt{ID} column should follow the format: \texttt{\{patient\_id\}\_\{joint\_index\}} (e.g., \texttt{1\_14} for the 14th joint of patient 1).
- \textbf{Bounding box coordinates (\texttt{xcenter}, \texttt{ycenter}, \texttt{dx}, \texttt{dy})} must be normalized to the image dimensions.

---

# Baseline Solution

As a baseline, we present a pipeline combining \textbf{YOLOv12s} for joint localization and \textbf{ResNet50} for pathology assessment.

## Training

### Preprocessing for Detection Model

\begin{verbatim}
chmod +x preprocess_for_yolo.sh
./preprocess_for_yolo.sh --csv /path/to/csv --img_dir /path/to/images --split_info_path /path/to/split_info.json
\end{verbatim}

### Train YOLO

1. Add your \texttt{yolo\_dataset} path to the detection config.
2. Run the training script:

\begin{verbatim}
python detection/model/yolo/train.py
\end{verbatim}

## Preprocessing Data for Classifier Training

The classifier (ResNet50) is trained on ground-truth bounding boxes. The preprocessing steps include:
- Averaging expert scores.
- Merging scores and bounding box files.
- Normalizing bounding box coordinates and cropping images.

\begin{verbatim}
chmod +x preprocess_for_classifier.sh
./preprocess_for_classifier.sh --scores_csv /path/to/scores.csv --bbox_file /path/to/bboxes.csv --image_dir /path/to/jpeg --split_subsets_by_id /path/to/train_val_split.json
\end{verbatim}

### Training the Classifier

1. Add your Weights \& Biases (wandb) key to \texttt{config/classifier/train.yaml}.
2. Configure the model in the config file (any model can be initialized via Hydra).
3. Run the training script:

\begin{verbatim}
python classifier/train.py
\end{verbatim}

---

## Evaluate and Submit

### Detection

1. Detect joints on evaluation data:

\begin{verbatim}
python detection/model/yolo/infer.py --model /path/to/model_weights --img_dir /path/to/eval_data --output_dir /path/to/output_dir --imgsz pretrained_model_image_size
\end{verbatim}

2. Crop images using the generated bounding boxes:

\begin{verbatim}
python data_utils/crop_eval.py --image_dir /path/to/image/jpeg --bbox_txt /path/to/detection/all_bboxes.txt --output_dir data/eval_cropped_images
\end{verbatim}

### Classification

1. Predict labels for cropped images (edit \texttt{config/classification/submit.yaml}):
   - \texttt{model\_weights}: Path to the best classification model checkpoint.
   - \texttt{bbox\_csv}: Path to predicted bounding boxes from the detection model.
   - \texttt{image\_dir}: Path to cropped images (\texttt{data/eval\_cropped\_images}).
   - \texttt{output\_csv}: Path to save the submission file (\texttt{submit.csv}).

2. Run the submission script:

\begin{verbatim}
python submit.py
\end{verbatim}

---

## Submission File Requirements

- Ensure the \texttt{submit.csv} file adheres to the specified format.
- Normalize bounding box coordinates to the image dimensions.
- Include exactly 100 rows per image, padding with empty rows if necessary.
- Use the \texttt{PAD} column to indicate padding rows (\texttt{1} for padding, \texttt{0} for predicted rows).
