# Training

## Preprocessing for detection model

```bash
chmod +x preprocess_for_yolo.sh
./preprocess_for_yolo.sh --csv /path/to/csv --img_dir /path/to/images --split_info_path /path/to/split_info.json
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
chmode +x preprocess_for_classifier.sh
./preprocess_for_classifier.sh --scores_csv /path/to/scores.csv --bbox_file /path/to/bboxes.csv --image_dir /path/to/jpeg --split_subsets_by_id /path/to/train_val_split.json
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

``python
python data_utils/crop_eval.py --path/to/image/jpeg --bbox_txt path/to/detection/all_bboxes.txt --output_dir data/eval_cropped_images
```

## Classification

* predict labels for cropped images (edit config/classification/submit.yaml!!!)

config/classification/submit.yaml

  model_weights - path to best classification model checkpoint
  bbox_csv - path to predicted by detection model bboxes
  image_dir: - path to cropped images (data/eval_cropped_images)
  output_csv: - submit.csv

```python
python submit.py
```
