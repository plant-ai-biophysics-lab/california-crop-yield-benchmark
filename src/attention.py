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
from models.layers_ours import *
from transformers import BertTokenizer, BertModel
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel

from models.configs import set_seed
set_seed(1987)

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

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class RegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # reshape 
        self.norm =nn.LayerNorm(config.embed_dim*2)
        self.lfc = nn.Linear(self.config.embed_dim*2, 256, bias=True)  
        self.fcn = nn.GELU()
        self.fold = torch.nn.Fold(output_size=(self.config.img_size, self.config.img_size),
                                  kernel_size=1, dilation=1,
                                  padding=0, stride=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.lfc(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], 1, int(self.config.img_size**2))
        x = self.fold(x)

        return x

class MultiRegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.mask_modality != 'text': #
            n = 2
        else:
            n = 1

        self.regression = nn.ModuleDict(dict(
                norm =nn.LayerNorm(self.config.embed_dim*2), 
                lfc = nn.Linear(self.config.embed_dim*2, 256, bias=True),
                fcn = nn.GELU(),
                fold = torch.nn.Fold(output_size=(16, 16),
                                  kernel_size=1, dilation=1,
                                  padding=0, stride=1)
            ))

    def forward(self, x_list):

        out = []
        for x in x_list:
            x = self.regression.norm(x)
            x = self.regression.lfc(x)
            x = self.regression.fcn(x)
            x = x.view(x.shape[0], 1, int(256))
            x = self.regression.fold(x)

            out.append(x)

        return out

class SingleRegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.mask_modality != 'text': #
            n = 2
        else:
            n = 1

        self.regression = nn.ModuleDict(dict(
                norm =nn.LayerNorm(self.config.embed_dim*n), 
                lfc = nn.Linear(self.config.embed_dim*n, 256, bias=True),
                fcn = nn.GELU(),
                fold = torch.nn.Fold(output_size=(self.config.img_size, self.config.img_size),
                                  kernel_size=1, dilation=1,
                                  padding=0, stride=1)
            ))

    def forward(self, x):

        x = self.regression.norm(x)
        x = self.regression.lfc(x)
        x = self.regression.fcn(x)
        x = x.view(x.shape[0], 1, int(self.config.img_size**2))
        x = self.regression.fold(x)

        return x
    
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam
#========================================================================================#
#============================ Text Embedding and Encoder ================================#
#========================================================================================#
# class TextEmbed(nn.Module):
#     def __init__(self, 
#                  config: Union[Dict], 
#                  device=None):
#         super().__init__()
#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.device = device
#         self.config = config
        
#         self.TextEncoder = tiktoken.get_encoding('p50k_base')
#         self.proj = Linear(1, config['context_dim'])
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, config['context_dim']))
#         self.dropout = Dropout(config['proj_dropout'])
#         self.norm = nn.LayerNorm(config['context_dim'])

#     def forward(self, texts):
#         encoded_texts = [self.TextEncoder.encode(text) for text in texts]
#         max_length = 249 
#         padded_texts = [text[:max_length] + [0] * (max_length - len(text)) for text in encoded_texts]

#         # Create attention mask: 1 for actual tokens, 0 for padding
#         attention_mask = torch.tensor([[1] * len(text) + [0] * (max_length - len(text)) for text in encoded_texts], 
#                                       dtype=torch.float32).to(self.device).bool()

#         texts_tensor = torch.tensor(padded_texts, dtype=torch.float32).to(self.device)
#         texts_tensor = torch.unsqueeze(texts_tensor, dim=-1)

#         B = texts_tensor.shape[0]
        
#         cls_tokens = self.cls_token.expand(B, -1, -1)
        
#         texts_tensor = self.proj(texts_tensor)
#         texts_tensor = self.norm(texts_tensor)
        
#         texts_tensor = torch.cat((cls_tokens, texts_tensor), dim=1)

#         embeddings = self.dropout(texts_tensor)

#         return embeddings, attention_mask
    
# class GPTTextEmbedding(nn.Module):
#     def __init__(self, model_name='gpt2', max_length=250):
#         super(GPTTextEmbedding, self).__init__()
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  
#         self.model = GPT2Model.from_pretrained(model_name)
#         self.model.resize_token_embeddings(len(self.tokenizer))  
#         self.max_length = max_length

#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model.to(self.device)

#     def forward(self, text_batch):

#         tokens = self.tokenizer(text_batch, max_length=self.max_length, 
#                                 truncation=True, padding='max_length', 
#                                 return_tensors='pt')

#         input_ids = tokens['input_ids'].to(self.device)
#         attention_mask = tokens['attention_mask'].to(torch.bool).to(self.device)

#         outputs = self.model(input_ids=input_ids, 
#                              attention_mask=attention_mask)
     
#         token_embeddings = outputs.last_hidden_state
#         return token_embeddings, attention_mask
    
# class TextEmbedBERT(nn.Module):

#     def __init__(self, 
#                  config: Union[Dict], 
#                  device=None):
#         super().__init__()
#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.device = device
#         self.config = config
        
#         self.TextEncoder = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.BertModel = BertModel.from_pretrained('bert-base-uncased').to(self.device)
#         self.proj = Linear(self.BertModel.config.hidden_size, config['embed_dim'])
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim']))
#         self.dropout = Dropout(config['proj_dropout'])
#         self.norm = nn.LayerNorm(config.embed_dim)


#     def forward(self, texts):

#         # Tokenize and encode the texts
#         encoded_texts = self.TextEncoder(texts, padding='max_length', truncation=True, max_length=249, return_tensors='pt').to(self.device)

#         # Get the output embeddings from the BERT model
#         outputs = self.BertModel(**encoded_texts)
#         last_hidden_state = outputs.last_hidden_state


#         # Ensure fixed size output
#         B, T, C = last_hidden_state.shape
#         assert T == 249, "The tokenizer output length must be 249 tokens"
        
#         # Project the last hidden state to the desired embedding dimension
#         texts_tensor = self.proj(last_hidden_state)
#         texts_tensor = self.norm(texts_tensor)
#         # Add CLS token to the embeddings
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         texts_tensor = torch.cat((cls_tokens, texts_tensor), dim=1)  # Now the shape is (B, 250, embed_dim)
        
#         # Apply dropout
#         embeddings = self.dropout(texts_tensor)

#         return embeddings

# class TextEmbedBERT(nn.Module):

#     def __init__(self, output_dim=64):
#         super().__init__()
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#         self.bert_model = BertModel.from_pretrained('bert-base-cased')
        
#         # Linear layer to reduce embedding size from 768 to output_dim (default 64)
#         # self.projection_layer = nn.Linear(768, output_dim)

#     def forward(self, texts):
#         max_length = 255
#         inputs = self.tokenizer(
#             texts, 
#             return_tensors="pt", 
#             padding="max_length", 
#             truncation=True, 
#             max_length=max_length
#         ).to(device) 
        
#         self.bert_model.to(device)  
#         # self.projection_layer.to(device)  
#         with torch.no_grad():
#             outputs = self.bert_model(**inputs)
#             text_embeddings = outputs.last_hidden_state  
#         # projected_embeddings = self.projection_layer(text_embeddings)  

#         return text_embeddings
    
# class TextAttention(nn.Module):
#     def __init__(self, dim, num_heads = 8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5


#         self.matmul1 = einsum('bhid,bhjd->bhij')
#         self.matmul2 = einsum('bhij,bhjd->bhid')

#         self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = Dropout(attn_drop)
#         self.proj = Linear(dim, dim)
#         self.proj_drop = Dropout(proj_drop)
#         self.softmax = Softmax(dim=-1)

#         self.attn_cam = None
#         self.attn = None
#         self.v = None
#         self.v_cam = None
#         self.attn_gradients = None

#     def get_attn(self):
#         return self.attn

#     def save_attn(self, attn):
#         self.attn = attn

#     def save_attn_cam(self, cam):
#         self.attn_cam = cam

#     def get_attn_cam(self):
#         return self.attn_cam

#     def get_v(self):
#         return self.v

#     def save_v(self, v):
#         self.v = v

#     def save_v_cam(self, cam):
#         self.v_cam = cam

#     def get_v_cam(self):
#         return self.v_cam

#     def save_attn_gradients(self, attn_gradients):
#         self.attn_gradients = attn_gradients

#     def get_attn_gradients(self):
#         return self.attn_gradients

#     def forward(self, x, g):
#         b, n, _, h = *x.shape, self.num_heads
#         qkv = self.qkv(x)
#         q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

#         self.save_v(v)
#         dots = self.matmul1([q, k]) * self.scale
#         attn = self.softmax(dots)
#         attn = self.attn_drop(attn)

#         attn.requires_grad_(True)
#         self.save_attn(attn)
#         if g: 
#             attn.register_hook(self.save_attn_gradients)

#         out = self.matmul2([attn, v])
#         out = rearrange(out, 'b h n d -> b n (h d)')

#         out = self.proj(out)
#         out = self.proj_drop(out)

#         return out, attn

#     # def relprop(self, cam, **kwargs):
#     #     cam = self.proj_drop.relprop(cam, **kwargs)
#     #     cam = self.proj.relprop(cam, **kwargs)
#     #     cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)


#     #     (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
#     #     cam1 /= 2
#     #     cam_v /= 2

#     #     self.save_v_cam(cam_v)
#     #     self.save_attn_cam(cam1)

#     #     cam1 = self.attn_drop.relprop(cam1, **kwargs)
#     #     cam1 = self.softmax.relprop(cam1, **kwargs)

#     #     (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
#     #     cam_q /= 2
#     #     cam_k /= 2

#     #     cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

#     #     return self.qkv.relprop(cam_qkv, **kwargs)

class TextEmbed(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", max_length=250, device=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to the specified device
        self.model.to(self.device)

    def forward(self, texts):
        # Tokenize and move inputs to the specified device
        inputs = self.tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        attention_mask = inputs['attention_mask'].bool()

        # with torch.no_grad():
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        
        if embeddings.size(1) < self.max_length:
            pad_size = self.max_length - embeddings.size(1)
            padding = torch.zeros((embeddings.size(0), pad_size, embeddings.size(2))).to(self.device)
            embeddings = torch.cat([embeddings, padding], dim=1)

        return embeddings, attention_mask
    
class TextAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            # Expand mask to match attention scores dimensions
            mask = rearrange(mask, 'b n -> b 1 1 n')
            mask = mask.expand(b, h, n, n)  # Now mask is of shape [b, h, n, n]
            max_neg_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, max_neg_value)

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn

class TextEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = TextAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mult)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, out_features = dim, drop = dropout) 

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, text, mask):
        x1, x2 = self.clone1(text, 2)
        text, attn = self.attn(x2, mask)

        text = self.add1([x1, text])
        x1, x2 = self.clone2(text, 2)
        text = self.add2([x1, self.mlp(self.norm2(x2))])
        return text, attn

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam

#========================================================================================#
#========================================================================================#
#========================================================================================#
class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        # self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        B, T = x.shape[0], x.shape[-1]
        x = rearrange(x, 'b c w h t -> (b t) c w h')
        x = self.flattener(self.conv_layers(x)).transpose(-2, -1)
        x = rearrange(x, '(b t) n e -> b (t n) e', b=B)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class ImgEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 config):
        super().__init__()

        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if config.multi_conv:
            self.proj = nn.Sequential(
                nn.Conv2d(config.in_channels, config.embed_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.embed_dim // 4, config.embed_dim // 2, kernel_size=3, stride=2, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.embed_dim // 2, config.embed_dim, kernel_size=3, stride=2, padding = 1),
            )
        else:
            self.proj = nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size) #*temporal_obvs

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = nn.Dropout(config.proj_dropout)

    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)


        # x = rearrange(x, 'b c w h t -> b (c t) w h')
        x = self.proj(x).flatten(2).transpose(1, 2)

        x = torch.cat((cls_tokens, x), dim = 1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class STImgEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 config):
        super().__init__()

        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        temporal_obvs = 15
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * temporal_obvs
        
        self.proj = nn.Conv2d(config.in_channels * temporal_obvs, config.embed_dim, kernel_size=patch_size, stride=patch_size) 

        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = nn.Dropout(config.proj_dropout)

    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = rearrange(x, 'b c w h t -> b (c t) w h')
        x = self.proj(x).flatten(2).transpose(1, 2)

        x = torch.cat((cls_tokens, x), dim = 1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class AccImgPredEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, config, index):
        super().__init__()
        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (index)
        
        self.num_patches = num_patches
        self.proj_img = nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.proj_pred = nn.Conv2d(1, config.embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))   
        self.dropout = nn.Dropout(config.proj_dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x, pred):
        B, T = x.shape[0], x.shape[-1]
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = self.proj_img(x) 
        x = x.flatten(2).transpose(1, 2)  

        pred = self.proj_pred(pred)
        pred = pred.flatten(2).transpose(1, 2)

        x = torch.cat((pred, x), dim = 1)

        x = self.norm(x)
        # Concatenate CLS tokens and add position embeddings
        embeddings = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
class AccImgEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, config, index):
        super().__init__()
        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (index)
        
        self.num_patches = num_patches
        self.proj = nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))   
        self.dropout = nn.Dropout(config.proj_dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        B, T = x.shape[0], x.shape[-1]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand CLS tokens once
        # Rearrange to put time dimension next to batch
        x = rearrange(x, 'b c w h t -> (b t) c w h')
        # Apply convolution and adjust dimensions
        x = self.proj(x)  # Convolution output

        x = x.flatten(2).transpose(1, 2)  # Prepare for concatenation

        # Rearrange back into batches, handling non-contiguous layout with reshape
        x = rearrange(x, '(b t) n e -> b (t n) e', b=B)  # Reshape to original batch structure
        x = self.norm(x)
        # Concatenate CLS tokens and add position embeddings
        embeddings = torch.cat((cls_tokens, x), dim=1) + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

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

        # if exists(met):
        #     met = self.to_qkv(met).chunk(3, dim=-1)
        #     qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), met)
        #     met = einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
        #     dots += met
        
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class IncrementalSpatialAttention(nn.Module):
    def __init__(self, dim, heads=15, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # Removing the output projection to get individual head outputs
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        heads_outputs = []
        for head_idx in range(self.heads):
            # Each head processes an increasing window of tokens
            token_limit = 1 + ((head_idx + 1) * 4)  # Increasing window size for each head
            if token_limit > n:
                token_limit = n  # Ensure not to exceed number of tokens

            # Select tokens for this head
            qkv_head = [t[:, :token_limit] for t in qkv]
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_head)

            # Attention mechanism
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            # Append each head's output; we use nn.Identity() to skip any further projection
            out = self.to_out(out)

            
            heads_outputs.append(out)

        # Optionally, output could be rearranged or aggregated differently based on requirements
        return heads_outputs  # List of tensors, one per head

class IncrementalSpatialEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, IncrementalSpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def average_resize(self, tensor, target_length):
        """Resize the tensor to the target length using mean pooling."""
        tensor = tensor.transpose(1, 2)  
        resized_tensor = F.adaptive_avg_pool1d(tensor, target_length)
        resized_tensor = resized_tensor.transpose(1, 2)
        return resized_tensor

    def forward(self, x):
        
        for attn, ff in self.layers:
            attn_out = attn(x)
            x_ts = [x[:, :1 + ((index + 1) * 4), :] + attn_out[index] for index in range(len(attn_out))]
            x_ts = [ff(out) + out for out in x_ts]
            x_ts_avg = [output if idx == 0 else self.average_resize(output, 4) for idx, output in enumerate(x_ts)]
            x = torch.cat(x_ts_avg, dim=1)

        return [self.norm(out) for out in x_ts]
    
class SpatialEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SpatialMetAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, met):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(met):
            met = self.to_qkv(met).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), met)
            met = torch.einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
            dots += met
        
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn
    
class SpatialMetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        # self.drop_path = DropPath(0.05) if 0.05 > 0. else nn.Identity()


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialMetAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, met = None):
        attn_scores = []
        for attn, ff in self.layers:
            out, attn_score = attn(x = x, met = met)
            attn_scores.append(attn_score)
            x = out + x
            x = ff(x) + x

        return self.norm(x)#, attn_scores
    
#========================================================================================#
#========================================================================================#
#========================================================================================#

class YzEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 config, index):
        super().__init__()

        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (index)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(1, config.embed_dim, kernel_size=patch_size, stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches+1, config.embed_dim))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout             = nn.Dropout(config.proj_dropout)

    def forward(self, x):

        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = rearrange(x, 'b c w h t -> (b t) c w h')
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = rearrange(x, '(b t) n e -> b (t n) e', b=B)
        x = torch.cat((cls_tokens, x), dim = 1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class CrossAttnYZ(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, met = None, yz = None):
        h = self.heads
        
        if exists(yz):
            q = self.to_q(x)
            yz = default(yz, x)
            k = self.to_k(yz)
            v = self.to_v(yz)
        else:
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(met):
            q_met = self.to_q(met)
            k_met = self.to_k(met)
            q_met, k_met = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_met, k_met))

            met = torch.einsum('b i d, b j d -> b i j', q_met, k_met) * self.scale
            sim += met



        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)



        return self.to_out(out), attn
    
class SpatialMetYzAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, met, yz):
        b, n, _, h = *x.shape, self.heads

        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(met):
            met = self.to_qkv(met).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), met)
            met = torch.einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
            dots += met
        
        if exists(yz):
            qkv_yz = self.to_qkv(yz).chunk(3, dim=-1)
            qbyz, kbyz, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), qkv_yz)
            yz = torch.einsum('b h i d, b h j d -> b h i j', qbyz, kbyz) * self.scale
            dots += yz

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn

class SpatialMetYzEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttnYZ(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, met = None, yz = None):
        attn_scores = []
        for attn, ff in self.layers:
            out, attn_score = attn(x, met = met, yz = yz)
            attn_scores.append(attn_score)
            x = out + x
            x = ff(x) + x

        return self.norm(x), attn_scores
    
#========================================================================================#
#========================================================================================#
#========================================================================================#

class MetEmbed(nn.Module):
    """ Meteorological to Patch Embedding
    """
    def __init__(self, 
                 config,):
        super().__init__()


        self.num_patches = 15 

        self.proj = nn.Linear(4, config.embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches+1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.proj_dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

        
    def forward(self, met):
        B = met.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        met = rearrange(met, 'b c t d -> b t (c d)')
        met = self.proj(met)#.flatten(2).transpose(1, 2)
        met = self.norm(met)
        met = torch.cat((cls_tokens, met), dim = 1)
        embeddings = met + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class TempMetEmbed(nn.Module):
    """ 
    Meteorological to Patch Embedding.

    This module transforms meteorological data from a format [B, C, T, D] to a
    sequence of embeddings [B, 61, embed_dim], where:
    - B is the batch size
    - C is the number of channels (4)
    - T is the time dimension (15)
    - D is the depth of each feature (1)
    - 60 tokens are generated from the data, plus 1 class token.

    Parameters:
    - config: a configuration object with attributes embed_dim and proj_dropout.
    """
    def __init__(self, config):
        super().__init__()

        self.num_patches = 60  # Number of data-derived tokens
        self.proj = nn.Linear(4, config.embed_dim)  # Project C to embed_dim
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.proj_dropout)

    def forward(self, met):
        B = met.shape[0]
        # Expand class token to the batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Rearrange and project the input
        met = rearrange(met, 'b c t d -> (b t) c d')
        met = self.proj(met)  # Shape: (B*T, embed_dim)

        # Reshape back to (B, T, embed_dim) and add the class token
        met = rearrange(met, '(b t) d -> b t d', b=B)
        met = torch.cat((cls_tokens, met), dim=1)  # Shape: (B, 61, embed_dim)

        # Add position embeddings and apply dropout
        embeddings = met + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class AccMetEmbed(nn.Module):
    """ Metadata to Patch Embedding """
    def __init__(self, config, index):
        super().__init__()
        num_patches = 4 * (index)
        self.proj = nn.Linear(1, config.embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.proj_dropout)

    def forward(self, met):
        B = met.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Rearrange and process
        met = rearrange(met, 'b c t d -> b t (c d)')
        met = torch.unsqueeze(met, dim=3)  # Add a new dimension for linear projection
        met = self.proj(met)
        met = rearrange(met, 'b t n e -> b (t n) e')  # Flatten time and feature dimensions

        # Concatenate CLS tokens and position embeddings
        embeddings = torch.cat((cls_tokens, met), dim=1) + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
    
class MetAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

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
    
class MetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MetAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class IncrementalMetAttention(nn.Module):
    def __init__(self, dim, heads=15, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # Removing the output projection to get individual head outputs
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        heads_outputs = []
        for head_idx in range(self.heads):
            # Each head processes an increasing window of tokens
            token_limit = 1 + ((head_idx + 1) * 4)  # Increasing window size for each head
            if token_limit > n:
                token_limit = n  # Ensure not to exceed number of tokens

            # Select tokens for this head
            qkv_head = [t[:, :token_limit] for t in qkv]
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_head)

            # Attention mechanism
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            # Append each head's output; we use nn.Identity() to skip any further projection
            out = self.to_out(out)

            
            heads_outputs.append(out)

        # Optionally, output could be rearranged or aggregated differently based on requirements
        return heads_outputs  # List of tensors, one per head

class IncrementalMetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, IncrementalMetAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def average_resize(self, tensor, target_length):
        """Resize the tensor to the target length using mean pooling."""
        tensor = tensor.transpose(1, 2)  
        resized_tensor = F.adaptive_avg_pool1d(tensor, target_length)
        resized_tensor = resized_tensor.transpose(1, 2)
        return resized_tensor

    def forward(self, x):
        
        for attn, ff in self.layers:
            attn_out = attn(x)
            x_ts = [x[:, :1 + ((index + 1) * 4), :] + attn_out[index] for index in range(len(attn_out))]
            x_ts = [ff(out) + out for out in x_ts]
            x_ts_avg = [output if idx == 0 else self.average_resize(output, 4) for idx, output in enumerate(x_ts)]
            x = torch.cat(x_ts_avg, dim=1)

        return [self.norm(out) for out in x_ts]
   
#========================================================================================#
#========================================================================================#
#========================================================================================#
class MultiModalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # Also return the attention scores, reshaped back to [B, N, H, T] where N is query length and T is context length
        # attn = rearrange(attn, '(b h) n j -> b h n j', b=x.size(0), h=h)

        return self.to_out(out), attn

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, met = None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(met):
            met = self.to_qkv(met).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), met)
            met = torch.einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
            dots += met

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class MultiModalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, context_dim=9, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiModalAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head,
                                                 dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        attn_scores = []
        for attn, ff in self.layers:
            out, attn_score = attn(x, context=context)
            attn_scores.append(attn_score)
            x = out + x
            x = ff(x) + x
        return self.norm(x), attn_scores

class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, met=None):
        for attn, ff in self.layers:
            x = attn(x, met=met) + x
            x = ff(x) + x
        return self.norm(x)