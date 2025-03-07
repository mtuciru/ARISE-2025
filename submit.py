import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from classification.utils import load_eval_model
from data_utils.datasets import EvalImageDataset


def predict_scores(cfg, model, dataloader, device):
    results = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["image"])
            jsn_scores = outputs["jsn"].argmax(dim=1).cpu().numpy()  # Probability of positive class
            erosion_scores = outputs["erosion"].argmax(dim=1).cpu().numpy()  # Probability of positive class

            for i in range(len(batch["image"])):
                results.append({
                    "patient_id": int(batch["patient_id"][i].item()),
                    "joint_id": int(batch["joint_id"][i].item()),
                    "xcenter": batch["xcenter"][i].item(),
                    "ycenter": batch["ycenter"][i].item(),
                    "dx": batch["dx"][i].item(),
                    "dy": batch["dy"][i].item(),
                    "jsn_score": int(jsn_scores[i]),
                    "erosion_score": int(erosion_scores[i]),
                })
    return results

# Main function
@hydra.main(config_path="config/classification", config_name="submit")
def main(cfg: DictConfig):

    model = load_eval_model(cfg, cfg.inference.model_weights).to(cfg.inference.device)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    dataset = EvalImageDataset(cfg.inference.image_dir, cfg.inference.bbox_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size, shuffle=False)

    # Predict scores
    results = predict_scores(cfg, model, dataloader, cfg.inference.device)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.inference.output_csv, index=False)
    print(f"Predictions saved to {cfg.inference.output_csv}")

if __name__ == "__main__":
    main()
