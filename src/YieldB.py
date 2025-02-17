import torch
from torch import nn
from ml_collections import ConfigDict
from typing import Dict, Union
import os 
from timm.models.layers import trunc_normal_
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.configs import set_seed
set_seed(1987)

from src.attention import MultiModalEmbed, MMEncoder, MultiRegressionHead
device = "cuda" if torch.cuda.is_available() else "cpu"

class YieldBenchmark(nn.Module):

    def __init__(self, config: Union[Dict]): 
        super().__init__()

        if not isinstance(config, ConfigDict):
            raise ValueError("Config must be an instance of ml_collections.ConfigDict.")

        self.timeseries = config.timeseries

        if self.timeseries is True: 
            self.mm_embed = nn.ModuleList([MultiModalEmbed(config, i).to(device) for i in range(1, 13)])

        elif self.timeseries is False: 
            self.mm_embed = MultiModalEmbed(config, 12).to(device)

        self.encoder = MMEncoder(config)
        
        self.head = MultiRegressionHead(config)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        # Initialize Linear layers
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = 0.01)  # Truncated normal initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero

        # Initialize Conv2d layers
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming initialization for Conv2d
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero

        # Initialize LayerNorm layers
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # Set bias to zero
            nn.init.constant_(m.weight, 1.0)  # Set weights to 1

        # Initialize BatchNorm2d layers (if used)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)  # Set weights to 1 for BatchNorm
            nn.init.constant_(m.bias, 0)  # Set bias to zero for BatchNorm

        # # Optionally, initialize Embedding layers
        # elif isinstance(m, nn.Embedding):
        #     nn.init.normal_(m.weight, mean=0, std=0.01)  # Normal initialization for embeddings
    
    def single_month(self, 
                     landsat: torch.Tensor = None, 
                     et: torch.Tensor = None, 
                     climate: torch.Tensor = None, 
                     soil: torch.Tensor = None):
        
        l_embed , et_embed, climate_embed, soil_embed = self.mm_embed(landsat, et, climate, soil)

        x = torch.cat((l_embed , et_embed, climate_embed, soil_embed), dim = 1)
        out = []
        x = self.encoder(x = x)
        x = x.mean(dim = 1)
        out.append(x)
        return out

    def time_series(self,                      
                    landsat: torch.Tensor = None, 
                    et: torch.Tensor = None, 
                    climate: torch.Tensor = None, 
                    soil: torch.Tensor = None):


        out = []

        for index, mmt_embed in enumerate(self.mm_embed):
            landsat_t = landsat[:, :index+1, ...]
            et_t = et[:, :index+1, ...]
            climate_t = climate[:, :(index+1)*30, ...]

            l_embed , et_embed, climate_embed, soil_embed = mmt_embed(landsat_t, et_t, climate_t, soil)
            x = torch.cat((l_embed , et_embed, climate_embed, soil_embed), dim = 1)
            x = self.encoder(x = x)
            x = x.mean(dim = 1)
            out.append(x)
        
        return out
        
    def forward(self, 
                landsat: torch.Tensor = None, 
                et: torch.Tensor = None, 
                climate: torch.Tensor = None, 
                soil: torch.Tensor = None): 

        if self.timeseries is True:
            
            out = self.time_series(
                    landsat  = landsat, 
                    et = et, 
                    climate = climate, 
                    soil = soil)
            
        elif self.timeseries is False:
            out = self.single_month(
                    landsat  = landsat, 
                    et = et, 
                    climate = climate, 
                    soil = soil)

        preds = self.head(out)

        return preds


