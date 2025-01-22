import torch
from torch import nn
from ml_collections import ConfigDict
from typing import Dict, Union
from einops import rearrange, repeat
import os 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.configs import set_seed
set_seed(1987)

from models.attention import AccMetEmbed, AccImgEmbed, MultiRegressionHead, SpatialMetEncoder, TextEmbed, TextEncoder, MultiModalTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClimMgmtAware_ViT(nn.Module):

    def __init__(self, config: Union[Dict], cond: str = False): #
        super().__init__()
        self.cond = cond
        if not isinstance(config, ConfigDict):
            raise ValueError("Config must be an instance of ml_collections.ConfigDict.")
                
        self.text_embed = TextEmbed(model_name="distilbert-base-uncased", max_length=250)
        self.text_transformer = nn.ModuleList([TextEncoder(config.context_dim, 
            config.num_layers, 
            config.num_heads, 
            dim_head = 8, 
            mult=4, 
            dropout=config.proj_dropout)
            for i in range(config.num_layers)])

        self.timeseries = config.timeseries

        if config.timeseries is True: 
            self.img_embeds = nn.ModuleList([AccImgEmbed(config, i).to(device) for i in range(1, 16)])
            self.met_embeds = nn.ModuleList([AccMetEmbed(config, i).to(device) for i in range(1, 16)]) 

        elif self.timeseries is False: 
            self.img_embeds = AccImgEmbed(config, 15).to(device)
            self.met_embeds = AccMetEmbed(config, 15).to(device)

        self.spatialmet_encoder = SpatialMetEncoder(
            config.embed_dim,
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            mult=4, 
            dropout=config.proj_dropout)
        
        self.cross_attn_encoder = MultiModalTransformer(
            config.embed_dim, 
            config.num_layers, 
            config.num_heads, 
            dim_head = 96, 
            context_dim=config.context_dim, 
            mult=4,  
            dropout=config.proj_dropout)
        
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
    
    def single_week(self, img: torch.Tensor, context: str = None, met: torch.Tensor = None):
        context, mask = self.text_embed(context)
        block_attn, text_attn = [], []
        for txt_blk in self.text_transformer:
            context, attn = txt_blk(context, mask = mask) 
            block_attn.append(attn)
        block_attn = torch.stack(block_attn, dim=4) 
        text_attn.append(block_attn)
        context = context.mean(dim=1)
        context = torch.unsqueeze(context, dim=1)
        out = []
        img = self.img_embeds(img)
        met = self.met_embeds(met)
        ImgMet_t = self.spatialmet_encoder(x = img, met = met)
        ImgMet_t_mean = ImgMet_t[:, 1:].mean(dim=1)
        ImgMet_t_mean = torch.unsqueeze(ImgMet_t_mean, dim = 1)
        ImgMetText_t, _ = self.cross_attn_encoder(ImgMet_t_mean, context)
        ImgMetText_t = ImgMetText_t.mean(dim=1)
        ImgMetText_t = torch.unsqueeze(ImgMetText_t, dim = 1)
        ImgMetText_t = torch.cat((ImgMetText_t, ImgMet_t_mean), dim = 1)
        ImgMetText_t = ImgMetText_t.view(ImgMetText_t.size(0), -1)
        out.append(ImgMetText_t)

        return out, text_attn

    def time_series(self, img: torch.Tensor, context: str = None, met: torch.Tensor = None):

        context, mask = self.text_embed(context)
        block_attn, text_attn = [], []
        for txt_blk in self.text_transformer:
            context, attn = txt_blk(context, mask = mask) 
            block_attn.append(attn)
        
        block_attn = torch.stack(block_attn, dim=4) 
        text_attn.append(block_attn)

        context = context.mean(dim=1)
        context = torch.unsqueeze(context, dim=1)
        out = []

        for index, (img_embed, met_embed) in enumerate(zip(self.img_embeds, self.met_embeds)):

            img_t = img[..., :index+1]
            img_t = img_embed(img_t)
            met_t = met[:, :, :index+1, :]
            met_t = met_embed(met_t)
            ImgMet_t = self.spatialmet_encoder(x = img_t, met = met_t)

            ImgMet_t_mean = ImgMet_t[:, 1:].mean(dim=1)
            ImgMet_t_mean = torch.unsqueeze(ImgMet_t_mean, dim = 1)
            ImgMetText_t, _ = self.cross_attn_encoder(ImgMet_t_mean, context)
            ImgMetText_t = ImgMetText_t.mean(dim=1)
            ImgMetText_t = torch.unsqueeze(ImgMetText_t, dim = 1)

            ImgMetText_t = torch.cat((ImgMetText_t, ImgMet_t_mean), dim = 1)
            ImgMetText_t = ImgMetText_t.view(ImgMetText_t.size(0), -1)

            out.append(ImgMetText_t)
        
        return out, text_attn
        
    def forward(self, 
                img: torch.Tensor,
                context: str = None, 
                met: torch.Tensor = None,
                yz: torch.Tensor = None): 
        
        if self.cond is True: 
            img = img * yz

        
        if self.timeseries is True:
            out, text_attn = self.time_series(img = img, context = context, met = met)
        elif self.timeseries is False:
            out, text_attn = self.single_week(img = img, context = context, met = met)

        preds = self.head(out)

        return preds, text_attn


