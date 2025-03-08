# Computer Vision Hackathon: Automated Joint Assessment in Rheumatoid Arthritis

## Task Description

Rheumatoid arthritis (RA) is a chronic autoimmune disease characterized by inflammation, joint destruction, and extra-articular manifestations. Radiography is the standard imaging modality for diagnosing and monitoring joint damage in RA. However, traditional methods for evaluating radiographic progression, such as the Sharp method and its variants, are time-consuming and subjective. This hackathon focuses on developing automated solutions for joint assessment in RA using computer vision techniques.

Participants will build models to automatically score hand joints affected by RA. The task involves two key components:
1. **Joint Localization**: Accurately localize hand joints in radiographic images.
2. **Pathology Assessment**: Evaluate the severity of joint damage, specifically focusing on **erosion** and **joint space narrowing (JSN) and predict damage score (0-4 for JSN and 0-5 for erosion)**.

The goal is to develop a robust and efficient pipeline that can assist clinicians in diagnosing and monitoring RA progression, reducing subjectivity and manual effort.

---

## Evaluation Metrics

The performance of the models will be evaluated using the following metrics:

### 1. **Intersection over Union (IoU)**
IoU measures the overlap between the predicted bounding box and the ground truth bounding box. It is defined as:

$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$

- **Area of Overlap**: The region where the predicted and ground truth bounding boxes intersect.
- **Area of Union**: The total area covered by both the predicted and ground truth bounding boxes.

A higher IoU indicates better localization accuracy.

---



### 2.  Balanced Accuracy
Balanced Accuracy is a metric used to evaluate the performance of a classification model, especially in cases where the classes are imbalanced. It is the average of recall (sensitivity) obtained on each class, ensuring that the performance metric is not biased toward the majority class.

Balanced Accuracy is defined as:

$\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}} + \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}} \right)$

---

### 3. **Final Metric: IoU × Accuracy**
The final evaluation metric is the product of **IoU** and **Accuracy**. This combined metric ensures that models achieve both precise localization and accurate pathology assessment. It is defined as:

$\text{Final Metric} = \text{IoU} \times \text{Accuracy}$

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
# Baseline solution
As a baseline we present yolo12s + resnet50 pipeline.
## Training


## Preprocessing for detection model

  

```bash

chmod  +x  preprocess_for_yolo.sh

./preprocess_for_yolo.sh  --csv  /path/to/csv  --img_dir  /path/to/images  --split_info_path  /path/to/split_info.json

```

  

## Train yolo

  

* add your yolo_dataset path to detection config

```python

python detection/model/yolo/train.py

```

## Preprocessing data for classifier's training

  

Training classifier resnet50 on ground-truth bboxes (edit config/classification/train_dataset/val_dataset!!!)

  

* Average expert's scores

* Merge scores and bboxes files

* Normalize bboxes coordinates and crop images

  

Classifier will be trained on ground-truth bboxes (Detection model will be applied during evaluation)

  

```bash

chmod  +x  preprocess_for_classifier.sh

./preprocess_for_classifier.sh  --scores_csv  /path/to/scores.csv  --bbox_file  /path/to/bboxes.csv  --image_dir  /path/to/jpeg  --split_subsets_by_id  /path/to/train_val_split.json

```

  

## Training classifier

  

* add your wandb key to config/classifier/train.yaml

* add any model you want in config, it will be automatically initialized via hydra-core.

* run training script

  

```python

python classifier/train.py

```

  

# Evaluate and Submit

  
  

## Detection

  

* detection of joints on evaluation data

  

```python

python detection/model/yolo/infer.py --model path/to/model_weights --img_dir path/to/eval_data --output_dir path/to/output_dir --imgsz pretrained_model_image_size

```

  

* Cropping image by generated bboxes

  

```python

python data_utils/crop_eval.py --path/to/image/jpeg --bbox_txt path/to/detection/all_bboxes.txt --output_dir data/eval_cropped_images

```

  

## Classification

  

* predict labels for cropped images (edit config/classification/submit.yaml!!!)

  

__config/classification/submit.yaml__

  

* model_weights - path to best classification model checkpoint

* bbox_csv - path to predicted by detection model bboxes

* image_dir - path to cropped images (data/eval_cropped_images)

* output_csv - submit.csv

  

```python

python submit.py

```

