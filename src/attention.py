from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import einsum as en
from einops import rearrange, repeat
from torch.nn.modules.utils import _pair
import tiktoken
import os 
from typing import Union, Dict, List
from torch.nn.functional import interpolate

# from src.configs import set_seed
# set_seed(1987)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"



def exists(val):
    return val is not None

def uniq(arr):
    return {el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor



class LandsatEmbed(nn.Module):
    """ Image to Time-Series Token Embedding using Linear Projection """

    def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_channels = in_channels  
        self.index = time_steps  

        self.linear_proj = nn.Linear(self.in_channels*128, self.embed_dim)

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.index + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(proj_dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, C, N) where:
            B = batch size
            T = time steps
            C = spectral bands (channels)
            N = number of vectorized pixels

        Output: (B, Index * C, EmbedDim)
        """
        B, T, C, N = x.shape  
        # mask = x != -9999 
        # valid_x = torch.where(mask, x, torch.tensor(0.0, device=x.device))  
        # x = valid_x.mean(dim=-1, keepdim=True)  
        x = x.reshape(B * T, C*N)  
        x = self.linear_proj(x) 
        x = x.view(B, T, self.embed_dim)

        x = self.norm(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        x = self.dropout(x)

        return x  
    
class ETEmbed(nn.Module):
    """ Image to Time-Series Embedding using Linear Projection """

    def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_channels = in_channels  # Spectral bands (C)

        # Linear layer replacing Conv1D
        self.linear_proj = nn.Linear(self.in_channels*128, self.embed_dim)

        # Positional Encoding for Time-Series Representation
        self.position_embeddings = nn.Parameter(torch.zeros(1, time_steps + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(proj_dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, C, N) where:
            B = batch size
            T = time steps
            C = spectral bands (channels)
            N = variable number of vectorized pixels
        """

        B, T, C, N = x.shape  
        # mask = x != -9999 
        # valid_x = torch.where(mask, x, torch.tensor(0.0, device=x.device))  
        # x = valid_x.mean(dim=-1, keepdim=True)  
 
        x = x.permute(0, 1, 3, 2).reshape(B * T, C*N)  
        x = self.linear_proj(x) 
        x = x.view(B, T, self.embed_dim)
        x = self.norm(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        x = self.dropout(x)

        return x  
    
class SoilEmbed(nn.Module):
    """ Soil Data to Time-Series Embedding using Conv1D with Time Expansion """

    def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_channels = in_channels  #
        self.time_steps = time_steps  

        self.conv = nn.Conv1d(in_channels = 128, out_channels=self.embed_dim, kernel_size=1)

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.in_channels + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(proj_dropout)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        """
        x: (B, 1, C, N) where:
            B = batch size
            1 = single time step (soil is static)
            C = soil properties (features)
            N = number of vectorized soil samples
        """

        B, T, C, N = x.shape  
        # mask = x != -9999 
        # valid_x = torch.where(mask, x, torch.tensor(0.0, device=x.device))  
        # x = valid_x.mean(dim=-1, keepdim=True)  

        x = self.conv(x.view(B, N, C))  
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        x = self.dropout(x)

        return x  

class ClimateEmbed(nn.Module):
    """ Climate Data to Time-Series Embedding using Conv1D """

    def __init__(self, embed_dim, in_channels, time_steps, proj_dropout, downsample_factor):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_channels = in_channels  # Number of climate variables (C)
        self.time_steps = time_steps  # Number of months (1 to 12)
        self.downsample_factor = downsample_factor  # Downsample from 365 → time_steps

        # 1D Convolution for temporal downsampling per variable (C)
        self.conv = nn.Conv1d(
            in_channels=128,  # Single variable at a time
            out_channels=self.embed_dim, 
            kernel_size=self.downsample_factor, 
            stride=self.downsample_factor
        )

        # Positional Encoding for Time-Series Tokens
        self.position_embeddings = nn.Parameter(torch.zeros(1, (self.time_steps * self.in_channels) + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(proj_dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, 365, C, M) where:
            B = batch size
            365 = daily climate observations
            C = climate variables (features)
            M = number of spatial locations

        Output: (B, (C * time_steps) + 1, EmbedDim)
        """

        B, T, C, M = x.shape  
        # mask = x != -9999 
        # valid_x = torch.where(mask, x, torch.tensor(0.0, device=x.device))  
        # x = valid_x.mean(dim=-1, keepdim=True)  

        x = x.view(B, self.time_steps, self.downsample_factor, C, M)
        x = x.permute(0, 3, 1, 2, 4).reshape(B * C * self.time_steps, M, self.downsample_factor * 1)  # (B*C*T, 1, L)
        x = self.conv(x)  

        # Global Average Pooling over reduced sequence
        x = x.mean(dim=-1)  
        # Reshape back to (B, C, time_steps, EmbedDim)
        x = x.view(B, C, self.time_steps, self.embed_dim)
        # Flatten time and variables → (B, C*T, EmbedDim)
        x = x.permute(0, 2, 1, 3).reshape(B, C * self.time_steps, self.embed_dim)


        x = self.norm(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings

        x = self.dropout(x)

        return x 

class MultiModalEmbed(nn.Module):
    def __init__(self, config: Union[Dict], time_step: int):
        super().__init__()

        embed_dim = config.embed_dim
        landsat_channels = config.landsat_channels
        et_channels = config.et_channels
        climate_variables = config.climate_variables
        soil_variables = config.soil_variables
        proj_dropout = config.proj_dropout

        self.landsat = LandsatEmbed(
            embed_dim = embed_dim, 
            in_channels = landsat_channels, 
            time_steps = time_step, 
            proj_dropout = proj_dropout)
        
        self.et = ETEmbed(
            embed_dim = embed_dim, 
            in_channels = et_channels, 
            time_steps = time_step, 
            proj_dropout = proj_dropout)
        
        self.climate = ClimateEmbed(
            embed_dim = embed_dim, 
            in_channels = climate_variables, 
            time_steps = time_step, 
            proj_dropout = proj_dropout, 
            downsample_factor = 30)
        
        self.soil = SoilEmbed(
            embed_dim = embed_dim, 
            in_channels = soil_variables, 
            time_steps = time_step, 
            proj_dropout = proj_dropout)

    def forward(self, landsat, et, climate, soil):
        l_embed = self.landsat(landsat)  
        et_embed = self.et(et)  
        s_embed = self.soil(soil)  
        c_embed = self.climate(climate) 

        return l_embed, et_embed, c_embed, s_embed

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if not glu:
            self.project_in = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU()
            )
        else:
            self.project_in = GEGLU(dim, inner_dim)  
        
        self.dropout = nn.Dropout(dropout)
        self.project_out = nn.Linear(inner_dim, dim_out)
        
        self.net = nn.Sequential(
            self.project_in,
            self.dropout,
            self.project_out
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if isinstance(x, list):
            # Apply normalization and function fn to each tensor in the list
            return [self.fn(self.norm(xi)) for xi in x]
        else:
            # Single tensor processing as usual
            return self.fn(self.norm(x), **kwargs)

class MMAttention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out
    
class MMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.embed_dim
        depth = config.num_layers
        heads = config.num_heads
        head_dim = config.head_dim
        mult = 4
        dropout = config.attn_dropout

        
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        # self.drop_path = DropPath(0.05) if 0.05 > 0. else nn.Identity()


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MMAttention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class MultiRegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.regression = nn.ModuleDict(dict(
                norm =nn.LayerNorm(self.config.embed_dim), 
                lfc = nn.Linear(self.config.embed_dim, 1, bias=True),
            ))

    def forward(self, x_list):

        out = []
        for x in x_list:

            x = self.regression.norm(x)
            x = self.regression.lfc(x)

            out.append(x)
        return out
