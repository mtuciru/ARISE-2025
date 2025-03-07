import torch
import torchvision

from torch import nn
from torchvision import models


class ResNet_Model(nn.Module):
    def __init__(self,
                 model_name="resnet18",
                 n_output_layers=2,
                 output_layers_shapes=
                    {
                        5: "erosion",
                        4: "jsn"
                    },
                 pretrained=True):
        super().__init__()

        self.output_mapping = output_layers_shapes
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.backbone.fc.in_features, key) for key in self.output_mapping.keys()
            ]
        )
        
        self.backbone.fc = nn.Identity()
    
        
    def forward(self, x):
        feats = self.backbone(x)
        head_outputs = {
            self.output_mapping[self.heads[i].out_features]: self.heads[i](feats)
            for i in range(len(self.heads))
        }
        return head_outputs


if __name__ == "__main__":
    model = ResNet_Model()
    print(model(torch.randn((1, 3, 224, 224))))