import torch
import ml_collections
import random 
import numpy as np

class Configs():
    def __init__(self, 
                 embed_dim: int, 
                 landsat_channels: int, 
                 et_channels: int, 
                 climate_variables: int, 
                 soil_variables: int, 
                 num_heads: int, 
                 num_layers: int,
                 attn_dropout: float, 
                 proj_dropout: float, 
                 timeseries: str,
                 pool:str):


        self.embed_dim = embed_dim
        self.landsat_channels = landsat_channels
        self.et_channels = et_channels
        self.climate_variables = climate_variables
        self.soil_variables = soil_variables
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.timeseries = to_bool(timeseries)
        self.pool = pool
    
    def call(self):
        """Returns the Spatio-temporal configuration."""
        config = ml_collections.ConfigDict()

        config.embed_dim = self.embed_dim
        config.head_dim = int(self.embed_dim / self.num_heads)
        config.landsat_channels = self.landsat_channels
        config.et_channels = self.et_channels
        config.climate_variables = self.climate_variables
        config.soil_variables = self.soil_variables
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attn_dropout = self.attn_dropout
        config.proj_dropout = self.proj_dropout
        config.timeseries = self.timeseries
        config.pool = self.pool
        return config
   
def set_seed(seed=1987):
    """
    Set the seed for reproducibility across random, numpy, and torch (CPU and CUDA).
    
    Parameters:
    seed (int): The seed value to use for random, numpy, and PyTorch operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_bool(value):
    """
    Convert a string or any value to a boolean.
    
    If the value is a string, it is checked for 'true' or 'false' (case-insensitive).
    Otherwise, the value is cast to a boolean.
    
    Parameters:
    value (str or any): The value to convert to a boolean.
    
    Returns:
    bool: The converted boolean value.
    """
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
    return bool(value)