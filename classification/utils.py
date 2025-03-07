import torch
import os
from hydra.utils import instantiate

def save_checkpoint(model, epoch, jsn_accuracy, erosion_accuracy, save_dir):
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, f"model_epoch:{epoch}_jsn_accuracy:{jsn_accuracy}_erosion_accuracy:{erosion_accuracy}.pth") 
    )


def load_eval_model(cfg, model_weights_path):
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    return model