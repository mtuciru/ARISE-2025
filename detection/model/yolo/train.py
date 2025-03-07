import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

@hydra.main(config_path="../../../config/detection", config_name="train_yolo")
def train(cfg: DictConfig):
    model = YOLO(cfg.model.name)

    res = model.train(
        data=cfg.train.dataset_cfg,
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        imgsz=cfg.train.imgsz,
        device=cfg.train.device,
        workers=cfg.train.workers,
        patience=cfg.train.patience,
        save=cfg.train.save,
        save_period=cfg.train.save_period
    )

if __name__ == "__main__":
    train()
