import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
ModuleType = Union[str, Callable[..., nn.Module]]
from torch import Tensor, nn
# Step 1: Load the Pretrained Model
# For example, let's use a pretrained ResNet model from torchvision
from torchvision import models
import torch.nn.functional as F
import numpy as np
from augmentations import embed_data_mask
from models.model import *


def cal_correct(model_with_head,model,opt, data, device):
    vision_dset = opt.vision_dset
    x_categ, x_cont,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
    x_categ_c, c_cat = mask_with_0(x_categ, 0.2)
    x_cont_c, c_con = mask_with_0(x_cont, 0.2)
    x_categ_c=x_categ_c.long()
    _ , x_categ_enc_c, x_cont_enc_c = embed_data_mask(x_categ_c, x_cont_c, cat_mask, con_mask,model,vision_dset)
    x_categ_enc_c.to(device)
    x_cont_enc_c.to(device)
    output=model_with_head(x_categ_enc_c, x_cont_enc_c)
    cat = output["cat_out"]
                
    c_indecies = torch.nonzero(c_cat)
                
                
    correct=0
    total = 0
    for idxes in c_indecies:
        list_idx = idxes[1]
        tensor_idx = idxes[0]
        sp_out_cat=output["cat_out"][list_idx]
        sp_out_cat=sp_out_cat[tensor_idx,:]
        pre_result = torch.argmax(sp_out_cat).item()
        ground_truth = x_categ[idxes[0], idxes[1]].item()
        if pre_result == ground_truth:
            correct=correct+1
            total = total+1
        else: 
            total = total+1
    return correct, total

def mask_with_0(x,p):
    mask = torch.bernoulli(p * torch.ones(x.shape)).to(x.device)
    masked_input = torch.mul(1 - mask, x)
    masked_input = masked_input
    return masked_input, mask


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class GEGLU(nn.Module):
    """
    The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

class ReGLU(nn.Module):
    """
    The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == "reglu":
            return ReGLU()
        elif module_type == "geglu":
            return GEGLU()
        elif module_type == "gelu":
            return nn.GELU()
        elif module_type == "relu":
            return nn.ReLU()
        elif module_type == "leaky_relu":
            return nn.LeakyReLU()
        elif module_type == "layer_norm":
            return nn.LayerNorm(*args)
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(f"Failed to construct the module {module_type} with the arguments {args}") from err
            return cls(*args)
    else:
        return module_type(*args)

class ReconstructionHead(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            n_num_features: Optional[int] = 0,
            category_sizes: Optional[List[int]] = None,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_in, bias)

            self.num_out = nn.ModuleList([nn.Linear(d_in, 1) for _ in range(n_num_features)])

            if category_sizes:
                self.cat_out = nn.ModuleList([nn.Linear(d_in, o) for o in category_sizes])
            else:
                self.cat_out = None
            self.category_sizes = category_sizes

        def forward(self, x: Tensor):
            #x = x[:, :-1]
            x = self.linear(x)
            x = self.normalization(x)
            x = self.activation(x)

            if self.cat_out:
                x_cat = x[:, : len(self.category_sizes), :]
                cat_out = [f(x_cat[:, i]) for i, f in enumerate(self.cat_out)]
            else:
                cat_out = None

            
            x_num = x
            if self.category_sizes:
                x_num = x[:, len(self.category_sizes) :, :]
                
            
            num_out = [f(x_num[:, i]) for i, f in enumerate(self.num_out)]
            if len(num_out)>0:
                num_out = torch.cat(num_out, dim=1)
            else:
                num_out = None
            return {"num_out": num_out, "cat_out": cat_out}


class preprocess_net():
    def __init__(self, categories, num_continuous, d_token):
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dim = d_token
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
    def forward():
        pass

# Step 4: Create a Composite Model
class SAINT_Reconstruction(nn.Module):
    def __init__(self, base_model, new_head):
        super(SAINT_Reconstruction, self).__init__()
        self.base_model = base_model
        self.new_head = new_head

    def forward(self, x_categ_enc, x_cont_enc):
        x_base = self.base_model.transformer(x_categ_enc, x_cont_enc)
        x = self.new_head(x_base)
        return x
    
class FT_Reconstruction(nn.Module):
    def __init__(self, categories, num_continuous, d_token, num_special_tokens, base_model, new_head, mlp_act=None, dim_out = 1, mlp_hidden_mults = (4, 2), cont_embedding = "MLP"):
        super(FT_Reconstruction, self).__init__()
        self.cont_embeddings = cont_embedding
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.num_unique_categories = sum(categories)
        self.total_tokens = self.num_unique_categories + num_special_tokens
        self.dim = d_token
        
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)
        
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        
        # input_size = (d_token * self.num_categories)  + (d_token * num_continuous)
        # l = input_size // 8
        # hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        # all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
        # self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)
        
        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.base_model = base_model
        self.new_head = new_head

    def forward(self, x_categ_enc):
        x_base = self.base_model(x_categ_enc)
        x = self.new_head(x_base)
        return x
