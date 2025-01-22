import torch.nn as nn
import torchvision.models as models

from dataset.exceptions import InvalidBackboneError


# class ResNetSimCLR(nn.Module):

#     def __init__(self, base_model, out_dim):
#         super(ResNetSimCLR, self).__init__()
#         self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
#                             "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

#         self.backbone = self._get_basemodel(base_model)
#         dim_mlp = self.backbone.fc.in_features

#         # add mlp projection head
#         self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

#     def _get_basemodel(self, model_name):
#         try:
#             model = self.resnet_dict[model_name]
#         except KeyError:
#             raise InvalidBackboneError(
#                 "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
#         else:
#             return model

#     def forward(self, x):
#         return self.backbone(x)
    

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, in_channels=4):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)
        }

        self.backbone = self._get_basemodel(base_model)

        # Modify the first convolutional layer to accept `in_channels` channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        dim_mlp = self.backbone.fc.in_features

        # Add MLP projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Choose one of: resnet18 or resnet50"
            )
        else:
            return model

    def forward(self, x):
        return self.backbone(x)