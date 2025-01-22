import torch
import ml_collections
import random 
import numpy as np


blocks_information = {'LIV_003':['MALVASIA_BIANCA', '7', '12', '7', '1991', '4WIREWO', '1'], 
          'LIV_004':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_005':['CABERNET_SAUVIGNON', '2', '11', '5', '1996', 'LIVDC', '3'], 
          'LIV_006':['MALVASIA_BIANCA', '7', '12', '7', '1993', '4WIREWO', '1'], 
          'LIV_007':['SYMPHONY', '14', '10', '5', '1996', 'LIVDC', '3'], 
          'LIV_008':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_009':['PINOT_GRIS', '11', '10', '4', '2014', 'STACKEDT', '5'], 
          'LIV_010':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_011':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_012':['SYRAH', '15', '10', '8', '1995', 'LIVDC', '3'], 
          'LIV_013':['SYRAH', '15', '12', '7', '1995', 'LIVDC', '3'], 
          'LIV_014':['RIESLING', '13', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_015':['MALVASIA_BIANCA', '7', '12', '7', '1985', 'QUAD', '6'], 
          'LIV_016':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_017':['CABERNET_SAUVIGNON', '2', '11', '5', '1996', 'LIVDC', '3'], 
          'LIV_018':['CHARDONNAY', '3','10', '4', '1995', 'LIVDC', '3'], 
          'LIV_019':['RIESLING','13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_021':['PINOT_GRIS', '11', '10', '4', '2015', 'STACKEDT', '5'], 
          'LIV_022':['PINOT_NOIR', '12', '10', '5', '1997', 'SPLIT', '2'],
          'LIV_025':['CABERNET_SAUVIGNON', '2', '9', '9', '1996', '4WIREWO', '1'], 
          'LIV_026':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_027':['MERLOT', '8', '10', '4', '1994', 'LIVDC', '3'], 
          'LIV_028':['MUSCAT_CANELLI', '9', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_032':['CABERNET_SAUVIGNON', '2', '10', '5', '1996', 'LIVDC', '3'], 
          'LIV_038':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_050':['SYMPHONY', '14', '10', '7', '1997', 'LIVDC', '3'], 
          'LIV_058':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_061':['PINOT_GRIS', '11', '10', '6', '2002', '4WIREWO', '1'], 
          'LIV_062':['SYRAH', '15', '10', '8', '1995', 'LIVDC', '3'], 
          'LIV_063':['PINOT_GRIS', '11', '10', '4', '2014', 'STACKEDT', '5'], 
          'LIV_064':['CABERNET_SAUVIGNON', '2', '8', '9', '1997', '4WIREWO', '1'], 
          'LIV_066':['PINOT_GRIS', '11', '10', '4', '2012', 'SPLIT', '2'], 
          'LIV_068':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_070':['CABERNET_SAUVIGNON', '2', '8', '9', '1996', '4WIREWO', '1'], 
          'LIV_073':['PINOT_NOIR', '12', '12', '7', '2003', 'SPLIT', '2'], 
          'LIV_076':['RIESLING', '13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_077':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_089':['CHARDONNAY', '3', '9', '6', '2010', 'VERTICAL', '7'], 
          'LIV_090':['CHARDONNAY', '3', '11', '6', '1993', 'VERTICAL', '7'], 
          'LIV_094':['RIESLING', '13', '11', '5', '2014', 'STACKEDT', '5'], 
          'LIV_102':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_103':['CABERNET_SAUVIGNON', '2', '9', '6', '2012', 'HIGHWIRE', '8'], 
          'LIV_105':['MUSCAT_OF_ALEXANDRIA', '10', '10', '4', '2011', 'LIVDC', '3'], 
          'LIV_107':['RIESLING', '13', '11', '5', '2013', 'SPLIT', '2'], 
          'LIV_111':['CHARDONNAY', '3', '10', '4', '1995', 'LIVDC', '3'], 
          'LIV_114':['MALVASIA_BIANCA', '7', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_123':['MALBEC', '6', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_125':['DORNFELDER', '4', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_126':['CABERNET_SAUVIGNON', '2', '9', '6', '2010', 'TALLVERTICAL', '9'], 
          'LIV_128':['RIESLING', '13', '11', '5', '2013', 'SPLIT', '2'], 
          'LIV_135':['CHARDONNAY', '3', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_136':['ALICANTE_BOUSCHET', '1', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_163':['CHARDONNAY', '3', '12', '4', '1995', 'LIVDC', '3'], 
          'LIV_172':['MUSCAT_CANELLI', '9', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_175':['CHARDONNAY', '3', '11', '4', '1994', 'LIVDC', '3'], 
          'LIV_176':['CHARDONNAY', '3', '11', '7', '1994', '4WIREWO', '1'], 
          'LIV_177':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_178':['RIESLING', '13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_181':['SYMPHONY', '14', '10', '6', '1997', 'LIVDC', '3'], 
          'LIV_182':['LAMBRUSCO', '5', '11', '5', '2013', 'QUAD', '6'], 
          'LIV_186':['RIESLING', '13', '11', '7', '2012', 'SPLIT', '2'], 
          'LIV_193':['MERLOT', '8', '11', '5', '2010', 'SPLIT', '2']}

blocks_size = {'LIV_003_2016': [392, 1560],
 'LIV_003_2017': [394, 1560],
 'LIV_003_2018': [396, 1563],
 'LIV_003_2019': [392, 1563],
 'LIV_004_2016': [149, 779],
 'LIV_004_2017': [151, 778],
 'LIV_004_2018': [152, 782],
 'LIV_004_2019': [154, 780],
 'LIV_005_2016': [792, 998],
 'LIV_005_2017': [793, 998],
 'LIV_005_2018': [796, 1002],
 'LIV_006_2016': [436, 790],
 'LIV_006_2017': [437, 795],
 'LIV_006_2018': [442, 794],
 'LIV_006_2019': [439, 759],
 'LIV_007_2016': [788, 374],
 'LIV_007_2017': [789, 375],
 'LIV_007_2018': [792, 377],
 'LIV_007_2019': [791, 378],
 'LIV_008_2016': [672, 790],
 'LIV_008_2017': [672, 791],
 'LIV_008_2018': [676, 791],
 'LIV_009_2016': [825, 760],
 'LIV_009_2017': [827, 759],
 'LIV_009_2018': [826, 766],
 'LIV_009_2019': [827, 765],
 'LIV_010_2016': [326, 866],
 'LIV_010_2017': [325, 866],
 'LIV_010_2018': [330, 871],
 'LIV_011_2016': [505, 973],
 'LIV_011_2017': [506, 974],
 'LIV_011_2018': [509, 974],
 'LIV_011_2019': [509, 965],
 'LIV_012_2016': [704, 708],
 'LIV_012_2017': [702, 707],
 'LIV_012_2018': [703, 707],
 'LIV_012_2019': [706, 705],
 'LIV_013_2016': [323, 693],
 'LIV_013_2017': [323, 663],
 'LIV_013_2018': [325, 695],
 'LIV_013_2019': [324, 628],
 'LIV_014_2016': [758, 764],
 'LIV_014_2017': [759, 765],
 'LIV_014_2018': [759, 768],
 'LIV_014_2019': [759, 769],
 'LIV_016_2016': [842, 918],
 'LIV_016_2017': [827, 919],
 'LIV_016_2018': [839, 920],
 'LIV_016_2019': [843, 918],
 'LIV_017_2016': [377, 201],
 'LIV_017_2017': [378, 202],
 'LIV_017_2018': [376, 205],
 'LIV_018_2016': [666, 1582],
 'LIV_018_2017': [667, 1582],
 'LIV_018_2018': [671, 1585],
 'LIV_018_2019': [632, 1586],
 'LIV_019_2016': [355, 388],
 'LIV_019_2017': [355, 392],
 'LIV_019_2018': [322, 398],
 'LIV_019_2019': [312, 385],
 'LIV_021_2017': [1280, 1783],
 'LIV_021_2018': [1597, 1790],
 'LIV_021_2019': [1599, 1785],
 'LIV_022_2016': [388, 393],
 'LIV_022_2017': [389, 393],
 'LIV_022_2018': [393, 394],
 'LIV_025_2016': [892, 790],
 'LIV_025_2017': [820, 790],
 'LIV_025_2018': [863, 795],
 'LIV_025_2019': [895, 794],
 'LIV_026_2016': [584, 379],
 'LIV_026_2017': [590, 381],
 'LIV_027_2016': [127, 381],
 'LIV_027_2017': [130, 381],
 'LIV_028_2016': [373, 784],
 'LIV_028_2017': [383, 785],
 'LIV_028_2018': [384, 787],
 'LIV_028_2019': [333, 788],
 'LIV_032_2016': [350, 727],
 'LIV_032_2017': [393, 733],
 'LIV_032_2018': [399, 729],
 'LIV_038_2016': [405, 797],
 'LIV_038_2017': [405, 798],
 'LIV_038_2018': [409, 799],
 'LIV_050_2016': [423, 800],
 'LIV_050_2017': [424, 801],
 'LIV_050_2018': [427, 801],
 'LIV_050_2019': [427, 800],
 'LIV_058_2016': [394, 797],
 'LIV_058_2017': [395, 797],
 'LIV_058_2018': [399, 801],
 'LIV_061_2016': [677, 506],
 'LIV_061_2017': [678, 503],
 'LIV_061_2018': [684, 510],
 'LIV_061_2019': [646, 509],
 'LIV_062_2016': [733, 806],
 'LIV_062_2017': [733, 805],
 'LIV_062_2018': [734, 806],
 'LIV_063_2016': [1252, 801],
 'LIV_063_2017': [1252, 802],
 'LIV_063_2018': [1167, 818],
 'LIV_063_2019': [1265, 817],
 'LIV_066_2016': [366, 791],
 'LIV_066_2017': [367, 795],
 'LIV_066_2019': [373, 797],
 'LIV_068_2016': [366, 791],
 'LIV_068_2017': [189, 781],
 'LIV_070_2016': [189, 797],
 'LIV_070_2017': [594, 782],
 'LIV_070_2018': [598, 788],
 'LIV_073_2016': [783, 601],
 'LIV_073_2017': [786, 601],
 'LIV_073_2018': [791, 605],
 'LIV_076_2016': [393, 1026],
 'LIV_076_2017': [396, 1031],
 'LIV_076_2018': [397, 1032],
 'LIV_076_2019': [396, 1031],
 'LIV_077_2016': [907, 458],
 'LIV_077_2017': [610, 457],
 'LIV_077_2018': [914, 463],
 'LIV_077_2019': [911, 462],
 'LIV_089_2016': [234, 325],
 'LIV_089_2017': [235, 327],
 'LIV_089_2018': [236, 329],
 'LIV_089_2019': [236, 330],
 'LIV_090_2016': [271, 790],
 'LIV_090_2017': [272, 791],
 'LIV_090_2018': [276, 791],
 'LIV_090_2019': [274, 726],
 'LIV_094_2016': [146, 179],
 'LIV_094_2017': [379, 252],
 'LIV_094_2018': [381, 256],
 'LIV_094_2019': [382, 246],
 'LIV_102_2016': [793, 536],
 'LIV_102_2017': [739, 535],
 'LIV_102_2018': [794, 537],
 'LIV_102_2019': [787, 526],
 'LIV_103_2016': [402, 1164],
 'LIV_103_2017': [400, 1163],
 'LIV_103_2018': [408, 1168],
 'LIV_103_2019': [408, 1168],
 'LIV_105_2016': [666, 786],
 'LIV_105_2017': [644, 784],
 'LIV_105_2018': [671, 791],
 'LIV_105_2019': [668, 789],
 'LIV_107_2016': [757, 402],
 'LIV_107_2017': [790, 403],
 'LIV_107_2018': [794, 407],
 'LIV_107_2019': [727, 404],
 'LIV_111_2016': [317, 498],
 'LIV_111_2017': [366, 497],
 'LIV_111_2018': [331, 454],
 'LIV_111_2019': [319, 446],
 'LIV_114_2016': [333, 333],
 'LIV_114_2017': [333, 335],
 'LIV_114_2018': [339, 336],
 'LIV_114_2019': [338, 338],
 'LIV_123_2016': [391, 759],
 'LIV_123_2017': [392, 761],
 'LIV_123_2018': [395, 764],
 'LIV_123_2019': [394, 765],
 'LIV_125_2016': [425, 387],
 'LIV_125_2017': [426, 386],
 'LIV_125_2018': [361, 388],
 'LIV_125_2019': [430, 391],
 'LIV_128_2016': [378, 790],
 'LIV_128_2017': [379, 790],
 'LIV_128_2018': [384, 795],
 'LIV_128_2019': [350, 790],
 'LIV_135_2016': [814, 718],
 'LIV_135_2017': [747, 716],
 'LIV_135_2018': [821, 722],
 'LIV_135_2019': [821, 723],
 'LIV_136_2016': [740, 863],
 'LIV_136_2017': [815, 864],
 'LIV_136_2018': [818, 869],
 'LIV_136_2019': [816, 868],
 'LIV_172_2016': [393, 740],
 'LIV_172_2017': [394, 740],
 'LIV_172_2018': [400, 743],
 'LIV_172_2019': [398, 743],
 'LIV_176_2016': [375, 177],
 'LIV_176_2017': [375, 177],
 'LIV_176_2018': [379, 172],
 'LIV_176_2019': [379, 174],
 'LIV_177_2016': [832, 374],
 'LIV_177_2017': [822, 320],
 'LIV_177_2018': [839, 374],
 'LIV_177_2019': [839, 373],
 'LIV_178_2016': [399, 781],
 'LIV_178_2017': [403, 780],
 'LIV_178_2018': [405, 784],
 'LIV_178_2019': [338, 781],
 'LIV_181_2016': [384, 188],
 'LIV_181_2017': [352, 186],
 'LIV_181_2018': [391, 191],
 'LIV_181_2019': [390, 192],
 'LIV_182_2016': [384, 391],
 'LIV_182_2017': [385, 392],
 'LIV_182_2018': [385, 396],
 'LIV_182_2019': [383, 395],
 'LIV_186_2016': [504, 787],
 'LIV_186_2017': [542, 784],
 'LIV_186_2018': [548, 789],
 'LIV_186_2019': [547, 791],
 'LIV_193_2016': [223, 324],
 'LIV_193_2017': [227, 326],
 'LIV_193_2018': [159, 295],
 'LIV_193_2019': [107, 288], }

class Configs():
    def __init__(self, 
                 img_size: int, 
                 patch_size: int, 
                 in_channels: int, 
                 out_channels: int, 
                 embed_dim: int, 
                 context_dim: int,
                 mlp_dim: int, 
                 pool:str, 
                 num_heads: int, 
                 num_layers: int,
                 attn_dropout: float, 
                 proj_dropout: float, 
                 timeseries: str = False,
                 cond : str = False,
                 mask_modality: str = 'text'):

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.mlp_dim = mlp_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.timeseries = to_bool(timeseries)
        self.cond = to_bool(cond)
        self.pool = pool
        self.mask_modality = mask_modality
    
    def call(self):
        """Returns the Spatio-temporal configuration."""
        config = ml_collections.ConfigDict()

        config.img_size = self.img_size
        config.patch_size = self.patch_size 
        config.embed_dim = self.embed_dim
        config.context_dim = self.context_dim
        config.mlp_dim = self.mlp_dim
        config.in_channels = self.in_channels
        config.out_channels= self.out_channels
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attn_dropout = self.attn_dropout
        config.proj_dropout = self.proj_dropout
        config.timeseries = self.timeseries
        config.pool = self.pool
        config.cond = self.cond
        config.mask_modality = self.mask_modality

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