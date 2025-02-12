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




# class LandsatEmbed(nn.Module):
#     """ Image to Time-Series Token Embedding using Conv1D """

#     def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.in_channels = in_channels  # Spectral bands (C)
#         self.index = time_steps  # Time index multiplier

#         # 1D Convolution to extract features for each spectral channel
#         self.conv = nn.Conv1d(in_channels = self.in_channels, out_channels=self.embed_dim, kernel_size=1)

#         # Positional Encoding for Time-Series Tokens
#         self.position_embeddings = nn.Parameter(torch.zeros(1, self.index + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dropout = nn.Dropout(proj_dropout)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         """
#         x: (B, T, C, N) where:
#             B = batch size
#             T = time steps
#             C = spectral bands (channels)
#             N = number of vectorized pixels

#         Output: (B, Index * C, EmbedDim)
#         """

#         B, T, C, N = x.shape  # Expecting (B, T, C, N)

#         # Reshape input: process each channel separately by flattening across batch and time
#         x = x.reshape(B * T, C, N)  # (B*T, C, N)

#         # Ensure 1D Conv input is in (B, C, L) format
#         x = self.conv(x)  # (B*C, EmbedDim, N)

#         # Global Average Pooling over N (spatial pixels)
#         x = x.mean(dim=-1)  # (B*T, EmbedDim)

#         # Reshape back to (B, C, EmbedDim)
#         x = x.view(B, T, self.embed_dim)
#         x = self.norm(x)


#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
#         x = self.dropout(x)

#         return x  # Output shape: (B, Index * C, EmbedDim)

class LandsatEmbed(nn.Module):
    """ Image to Time-Series Token Embedding using Linear Projection """

    def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_channels = in_channels  # Spectral bands (C)
        self.index = time_steps  # Time index multiplier

        # Replacing Conv1D with a Linear layer
        self.linear_proj = nn.Linear(self.in_channels, self.embed_dim)

        # Positional Encoding for Time-Series Tokens
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

        B, T, C, N = x.shape  # Expecting (B, T, C, N)

        # Reshape input: (B, T, C, N) → (B*T*N, C)
        x = x.permute(0, 1, 3, 2).reshape(B * T * N, C)  # (B*T*N, C)

        # Apply Linear Projection
        x = self.linear_proj(x)  # (B*T*N, EmbedDim)

        # Reshape back: (B, T, N, EmbedDim)
        x = x.view(B, T, N, self.embed_dim)

        # Global Average Pooling over N (spatial pixels)
        x = x.mean(dim=2)  # (B, T, EmbedDim)

        # Apply LayerNorm
        x = self.norm(x)

        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        x = self.dropout(x)

        return x  # Output shape: (B, Index * C, EmbedDim)
    
# class ETEmbed(nn.Module):
#     """ Image to Time-Series Embedding using Conv1D """

#     def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.in_channels = in_channels  # Spectral bands (C)

#         # 1D Convolution to extract features across spectral channels
#         self.conv = nn.Conv1d(in_channels = self.in_channels, out_channels=self.embed_dim, kernel_size=1)

#         # Positional Encoding for Time-Series Representation
#         self.position_embeddings = nn.Parameter(torch.zeros(1, time_steps + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dropout = nn.Dropout(proj_dropout)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         """
#         x: (B, T, C, N) where:
#             B = batch size
#             T = time steps
#             C = spectral bands (channels)
#             N = variable number of vectorized pixels
#         """

#         B, T, C, N = x.shape
#         x = x.reshape(B * T, C, N)  
#         x = self.conv(x)  # Merge B and T for independent processing
#         x = x.mean(dim=-1)  # Global Average Pooling over pixels (N) → (B * T, Embed)

#         # Reshape back to (B, T, Embed)
#         x = x.view(B, T, self.embed_dim)

#         x = self.norm(x)
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
#         x = self.dropout(x)

#         return x  # Output shape: (B, T, Embed)

class ETEmbed(nn.Module):
    """ Image to Time-Series Embedding using Linear Projection """

    def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_channels = in_channels  # Spectral bands (C)

        # Linear layer replacing Conv1D
        self.linear_proj = nn.Linear(self.in_channels, self.embed_dim)

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

        B, T, C, N = x.shape  # Expecting (B, T, C, N)

        # Reshape input: (B, T, C, N) → (B*T*N, C)
        x = x.permute(0, 1, 3, 2).reshape(B * T * N, C)  # (B*T*N, C)

        # Apply Linear Projection
        x = self.linear_proj(x)  # (B*T*N, EmbedDim)

        # Reshape back: (B, T, N, EmbedDim)
        x = x.view(B, T, N, self.embed_dim)

        # Global Average Pooling over N (spatial pixels)
        x = x.mean(dim=2)  # (B, T, EmbedDim)

        # Apply LayerNorm
        x = self.norm(x)

        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        x = self.dropout(x)

        return x  # Output shape: (B, T, EmbedDim)
    
class SoilEmbed(nn.Module):
    """ Soil Data to Time-Series Embedding using Conv1D with Time Expansion """

    def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_channels = 5  # Number of soil features (C)
        self.time_steps = time_steps  # Fixed to 12

        # 1D Convolution to extract features across soil properties
        self.conv = nn.Conv1d(in_channels = 1, out_channels=self.embed_dim, kernel_size=1)

        # Positional Encoding for Time-Series Representation
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

        B, T, C, N = x.shape  # Expecting T=1 initially

        x = self.conv(x.view(B * C, T, N))
        x = x.mean(dim=-1)  # Global Average Pooling over pixels (N) → (B * T, Embed)

        x = x.view(B, C, self.embed_dim)
        x = self.norm(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        x = self.dropout(x)

        return x  

# class SoilEmbed(nn.Module):
#     """ Soil Data to Time-Series Embedding using Linear Projection """

#     def __init__(self, embed_dim, in_channels, time_steps, proj_dropout):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.in_channels = in_channels  # Soil properties (C)
#         self.time_steps = time_steps  # Fixed to 12

#         # Linear projection instead of Conv1D
#         self.linear_proj = nn.Linear(self.in_channels, self.embed_dim)

#         # Positional Encoding for Time-Series Tokens
#         self.position_embeddings = nn.Parameter(torch.zeros(1, self.in_channels + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dropout = nn.Dropout(proj_dropout)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         """
#         x: (B, 1, C, N) where:
#             B = batch size
#             1 = single time step (soil is static)
#             C = soil properties (features)
#             N = number of vectorized soil samples
#         """

#         B, T, C, N = x.shape  # Expecting T=1 initially

#         # Ensure x maintains 4D shape
#         x = x.view(B, C, T, N)  # (B, C, 1, N)

#         # Reshape input: (B, C, 1, N) → (B*N, C)
#         x = x.permute(0, 2, 1, 3).reshape(B * N, C)  # (B*N, C)

#         # Apply Linear Projection
#         x = self.linear_proj(x)  # (B*N, EmbedDim)

#         # Reshape back: (B, N, EmbedDim)
#         x = x.view(B, N, self.embed_dim)

#         # Apply LayerNorm
#         x = self.norm(x)
#         print(f"soil 0: {x.shape}")
#         # Add CLS token and position embeddings
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
#         x = self.dropout(x)
#         print(f"soil: {x.shape}")
#         return x  # Output shape: (B, C+1, EmbedDim)

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
            in_channels=1,  # Single variable at a time
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

        B, T_full, C, M = x.shape  # Expecting T_full=365

        # Step 1: Keep only required time steps
        x = x[:, :self.time_steps * self.downsample_factor]  # (B, T_needed, C, M)

        # Step 2: Reshape for Conv1D (B, C, time_steps, days_per_month, M)
        x = x.view(B, self.time_steps, self.downsample_factor, C, M)
        
        # Step 3: Process each variable separately (B, C, time_steps, days_per_month, M)
        x = x.permute(0, 3, 1, 2, 4).reshape(B * C * self.time_steps, 1, self.downsample_factor * M)  # (B*C*T, 1, L)

        # Step 4: Apply Conv1D over time
        x = self.conv(x)  # (B*C*T, EmbedDim, L_reduced)

        # Step 5: Global Average Pooling over reduced sequence
        x = x.mean(dim=-1)  # (B*C*T, EmbedDim)

        # Step 6: Reshape back to (B, C, time_steps, EmbedDim)
        x = x.view(B, C, self.time_steps, self.embed_dim)

        # Step 7: Flatten time and variables → (B, C*T, EmbedDim)
        x = x.permute(0, 2, 1, 3).reshape(B, C * self.time_steps, self.embed_dim)

        # Step 8: Normalize
        x = self.norm(x)

        # Step 9: Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings

        # Step 10: Apply dropout
        x = self.dropout(x)

        return x  # Output shape: (B, (C * time_steps) + 1, EmbedDim)


# class ClimateEmbed(nn.Module):
#     """ Climate Data to Time-Series Embedding using Linear Projection """

#     def __init__(self, embed_dim, in_channels, time_steps, proj_dropout, downsample_factor):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.in_channels = in_channels  # Number of climate variables (C)
#         self.time_steps = time_steps  # Number of months (1 to 12)
#         self.downsample_factor = downsample_factor  # Downsample from 365 → time_steps

#         # Linear layer replacing Conv1D for temporal downsampling
#         self.linear_proj = nn.Linear(self.downsample_factor * in_channels, self.embed_dim)

#         # Positional Encoding for Time-Series Tokens
#         self.position_embeddings = nn.Parameter(torch.zeros(1, (self.time_steps * self.in_channels) + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dropout = nn.Dropout(proj_dropout)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         """
#         x: (B, 365, C, M) where:
#             B = batch size
#             365 = daily climate observations
#             C = climate variables (features)
#             M = number of spatial locations

#         Output: (B, (C * time_steps) + 1, EmbedDim)
#         """
#         print(f"original: {x.shape}")
#         B, T, C, M = x.shape  
#         x = x.view(B, self.time_steps, T, C, M)
#         print(f"0 : {x.shape}")
#         x = x.permute(0, 1, 4, 2, 3).reshape(B * C * self.time_steps, T * M)  
#         print(f" init: {x.shape}")

#         x = self.linear_proj(x)  

#         # x = x.view(B, C, self.time_steps, self.embed_dim)


#         # x = x.mean(dim=1)  # (B, time_steps, EmbedDim)


#         x = x.view(B, self.time_steps * C, self.embed_dim)


#         x = self.norm(x)


#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings


#         x = self.dropout(x)

#         return x 
    

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
