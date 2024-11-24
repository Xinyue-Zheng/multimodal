import torch
from torch import nn
from models import SAINT, SAINT_vision,SAINT_con, SAINT_dict
from models.model import *

import pandas as pd

from data_openml import DataSetCatCon, data_prep_df,DataSetCatCon_without_target, data_prep_keyvalue, DataSetCatCon_dict, DataSetCatCon_align
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
from pretraining import SAINT_pretrain,pretrain_process, pretrain_process_dict
from augmentations import embed_data_dict
#from CrossVLT import SegModel

import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', default = "1461", type=int)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', default = "binary", type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 1 , type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_noise_type', default=None , type=str,choices = ['missing','cutmix'])
parser.add_argument('--train_noise_level', default=0, type=float)

parser.add_argument('--ssl_samples', default= None, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])


opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Specify the dataset ID you want to download

df = pd.read_csv('bank_data.csv')
df = df.head(3654)
nan_percentage = 0.5
df.replace("unknown", np.nan, inplace=True)

cont_columns=['V6', 'V13', 'V1', 'V12', 'V15', 'V14', 'V10']

for column in cont_columns:
    nan_percentage = 0.1  # e.g., 10% of values

    # Calculate the number of values to replace
    num_nan = int(len(df) * nan_percentage)

    # Randomly select indices to replace with NaN
    nan_indices = df.sample(n=num_nan, random_state=1).index

    # Set the selected indices in the specified column to NaN
    df.loc[nan_indices, column] = np.nan



df_dict = df.to_dict(orient='records')

column_headers = list(df_dict[0].keys())
cat_columns = []
cont_columns = []
for column in df.columns:
    # Remove NaNs temporarily
    non_na_data = df[column].dropna()
    see = pd.api.types.is_numeric_dtype(non_na_data)
    # Check if column is numerical by attempting conversion
    if pd.api.types.is_numeric_dtype(non_na_data) or pd.to_numeric(non_na_data, errors='coerce').notna().all():
        cont_columns.append(column)
    else:
        cat_columns.append(column)

# cat_columns=['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16','Class']
# cont_columns=['V6', 'V13', 'V1', 'V12', 'V15', 'V14', 'V10']
start_date = "2015-01-01"
end_date = "2025-01-01"

# Generate random timestamps within the specified range
num_rows = len(df)
df['timestamp'] = pd.date_range(start=start_date, end=end_date, freq='D')
time_step_head = 'timestamp'
time_idx =17
time_idx_1 = 0

df2 = pd.DataFrame({
    "timestamp": df['timestamp'],
    "KPI1": np.random.uniform(0, 100, size=num_rows),  # Random numbers between 0 and 100
    "KPI2": np.random.uniform(-50, 50, size=num_rows),  # Random numbers between -50 and 50
    "KPI3": np.random.uniform(1, 10, size=num_rows),    # Random numbers between 1 and 10
    "KPI4": np.random.normal(0, 1, size=num_rows),      # Random numbers from a normal distribution
    "KPI5": np.random.randint(0, 1000, size=num_rows)   # Random integers between 0 and 1000
})
cat_dims, cat_idxs, con_idxs, key_dims, X_train, X_valid, X_test, train_mean, train_std = data_prep_df(df,cat_columns,cont_columns,column_headers,time_step_head, datasplit=[1,0,0])
cat_dims_1, cat_idxs_1, con_idxs_1, key_dims_1, X_train_1, X_valid_1, X_test_1, train_mean_1, train_std_1 = data_prep_df(df2,None,list(set(df2.columns.tolist())-set(["timestamp"])),df2.columns.tolist(),time_step_head, datasplit=[1,0,0])


#cat_dims, cat_idxs, con_idxs, key_dims, X_train, X_valid, X_test, train_mean, train_std = data_prep_keyvalue(df_dict,cat_columns,cont_columns, datasplit=[.65, .15, .2])
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

continuous_mean_std_1 = np.array([train_mean_1,train_std_1]).astype(np.float32) 

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(4,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = 4
    opt.attention_dropout = 0.8
    opt.embedding_size = 16
    if opt.optimizer =='SGD':
        opt.ff_dropout = 0.4
        opt.lr = 0.01
    else:
        opt.ff_dropout = 0.8


opt.pretrained_swin_weights = "pretrained/swin_base_patch4_window12_384_22k.pth"
# train_ds = DataSetCatCon_without_target(X_train, cat_idxs, continuous_mean_std)
# trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

# valid_ds = DataSetCatCon_without_target(X_valid, cat_idxs, continuous_mean_std)
# validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

# test_ds = DataSetCatCon_without_target(X_test, cat_idxs, continuous_mean_std)
# testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

train_ds_1 = DataSetCatCon_dict(X_train, cat_idxs, time_idx, continuous_mean_std)
trainloader_1 = DataLoader(train_ds_1, batch_size=opt.batchsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon_dict(X_valid, cat_idxs, time_idx, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

# test_ds = DataSetCatCon_dict(X_test, cat_idxs, time_idx, continuous_mean_std)
# testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)
train_ds = DataSetCatCon_align(X_train, X_train_1, cat_idxs, cat_idxs_1,time_idx, time_idx_1, continuous_mean_std, continuous_mean_std_1)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)



model = SAINT_dict(
key_dims=key_dims,
categories = tuple(cat_dims), 
num_continuous = len(con_idxs),                
dim = opt.embedding_size,                           
dim_out = 1,                       
depth = opt.transformer_depth,                       
heads = opt.attention_heads,                         
attn_dropout = opt.attention_dropout,             
ff_dropout = opt.ff_dropout,                  
mlp_hidden_mults = (4, 2),       
cont_embeddings = opt.cont_embeddings,
attentiontype = opt.attentiontype,
final_mlp_style = opt.final_mlp_style,
)
model.to(device)

model_1 = SAINT_dict(
key_dims=key_dims,
categories = tuple(cat_dims_1), 
num_continuous = len(con_idxs_1),                
dim = opt.embedding_size,                           
dim_out = 1,                       
depth = opt.transformer_depth,                       
heads = opt.attention_heads,                         
attn_dropout = opt.attention_dropout,             
ff_dropout = opt.ff_dropout,                  
mlp_hidden_mults = (4, 2),       
cont_embeddings = opt.cont_embeddings,
attentiontype = opt.attentiontype,
final_mlp_style = opt.final_mlp_style,
)
model_1.to(device)

# model_attn = SegModel(opt,
#                 pretrain_img_size=384,
#                 patch_size=4,
#                 embed_dim=128, 
#                 depths=[2, 2, 18, 2],
#                 num_heads=[4,8,16,32], 
#                 window_size=12,
#                 mlp_ratio=4.,
#                 qkv_bias=True,
#                 qk_scale=None,
#                 drop_rate=0.,
#                 attn_drop_rate=0.,
#                 drop_path_rate=0.3,
#                 norm_layer=nn.LayerNorm,
#                 patch_norm=True,
#                 use_checkpoint=False,
#                 training=False
#                 )
# model_attn.to(device)

#model_attn = Fusion_layer()




optimizer = optim.AdamW(model.parameters(),lr=0.0001)
window_size = 7
batch_size = 256
pt_aug_dict = {
    'noise_type' : opt.pt_aug,
    'lambda' : opt.pt_aug_lam
}
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
for epoch in range(opt.pretrain_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        
        x_categ, x_cont,cat_mask, con_mask, key = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
        
        #self attention + cross attention + align
        x_categ_1, x_cont_1,cat_mask_1, con_mask_1, key_1, time_stamp = data[5].to(device), data[6].to(device), data[7].to(device), data[8].to(device), data[9].to(device), data[10].to(device)

        
        if 'cutmix' in opt.pt_aug:
            from augmentations import add_noise
            x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
            _ , x_categ_enc_corr, x_cont_enc_corr = embed_data_dict(x_categ_corr, x_cont_corr, cat_mask, con_mask, model,key,time_stamp,vision_dset= False)
            
            x_categ_corr_1, x_cont_corr_1 = add_noise(x_categ_1,x_cont_1, noise_params = pt_aug_dict)
            _ , x_categ_enc_corr_1, x_cont_enc_corr_1 = embed_data_dict(x_categ_corr_1, x_cont_corr_1, cat_mask_1, con_mask_1, model,key,time_stamp,vision_dset= False)
        else: 
            _ , x_categ_enc_corr, x_cont_enc_corr = embed_data_dict(x_categ, x_cont, cat_mask, con_mask, model,key,time_stamp,vision_dset= False)
            _ , x_categ_enc_corr_1, x_cont_enc_corr_1 = embed_data_dict(x_categ_1, x_cont_1, cat_mask_1, con_mask_1, model_1,key_1,time_stamp,vision_dset= False)
        if 'mixup' in opt.pt_aug:
            from augmentations import mixup_data
            x_categ_enc_corr, x_cont_enc_corr = mixup_data(x_categ_enc_corr, x_cont_enc_corr , lam=opt.mixup_lam)
            x_categ_enc_corr_1, x_cont_enc_corr_1 = mixup_data(x_categ_enc_corr_1, x_cont_enc_corr_1 , lam=opt.mixup_lam)
        ds_corr_1 = torch.cat([x_categ_enc_corr, x_cont_enc_corr], dim = 1)
        ds_corr_1 = ds_corr_1.unsqueeze(1)
        
        enc_1_corr = torch.cat([x_categ_enc_corr_1, x_cont_enc_corr_1], dim = 1)
        
        
        
        _ , x_categ_enc, x_cont_enc = embed_data_dict(x_categ, x_cont, cat_mask, con_mask, model,key,time_stamp,vision_dset= False)
        ds_1 = torch.cat([x_categ_enc, x_cont_enc], dim = 1)
        ds_1 = ds_1.unsqueeze(1)
        
        _ , x_categ_enc_1, x_cont_enc_1 = embed_data_dict(x_categ_1, x_cont_1, cat_mask_1, con_mask_1, model_1,key_1,time_stamp,vision_dset= False)
        enc_1  = torch.cat([x_categ_enc_1, x_cont_enc_1], dim = 1)
        
        
        ds_2 = []
        ds_corr_2 = []
        x_cont_1_new = []
        for j in range(opt.batchsize):
            # Create a sliding window for each element in the batch
            window_start = max(0, j)
            window_end = min(batch_size, j + window_size)
            window = enc_1[window_start:window_end]  # Shape: [window_size, embed_dim]
            window_corr = enc_1_corr[window_start:window_end]
            window_cont = x_cont_1[window_start:window_end]
            x_cont_1_new.append(window_cont)
            ds_2.append(window)
            ds_corr_2.append(window_corr)
        
        max_window_length = window_size
        for k, window in enumerate(ds_2):
            padding = max_window_length - window.shape[0]
            if padding > 0:
                pad_tensor_enc = torch.zeros(padding,len(con_idxs_1)).to(device)
                pad_tensor = torch.zeros(padding,window.shape[1], opt.embedding_size).to(device)
                ds_2[k] = torch.cat([window, pad_tensor], dim=0)
                ds_corr_2[k] = torch.cat([ds_corr_2[k], pad_tensor], dim=0)
                x_cont_1_new[k] = torch.cat([x_cont_1_new[k], pad_tensor_enc], dim=0)
        x_cont_1_new = torch.stack(x_cont_1_new)
        x_cont_1_new = x_cont_1_new.view(opt.batchsize, -1)    
        
        # Stack sliding windows into a tensor
        ds_2 = torch.stack(ds_2)
        ds_corr_2 = torch.stack(ds_corr_2)
        
        ds_1_feature = model.transformer(ds_1)
        ds_1_corr_feature = model.transformer(ds_corr_1)
        
        ds_2_feature = model_1.transformer(ds_2)
        ds_2_corr_feature=model_1.transformer(ds_corr_2)
        
        ds_1_feature = model.cross_attention(ds_1_feature, ds_2_feature)
        ds_1_corr_feature = model.cross_attention(ds_1_corr_feature, ds_2_corr_feature)
        
        ds_2_feature = model_1.cross_attention(ds_2_feature, ds_1_feature)
        ds_2_corr_feature = model_1.cross_attention(ds_2_corr_feature, ds_1_corr_feature)
        
        
        
        #Reconstruction head
        
        ds_1_cat = model.mlp1(ds_1_corr_feature[:,:,1:len(cat_idxs)*2:2,:])
        ds_1_con = model.mlp2(ds_1_corr_feature[:,:,(len(cat_idxs)*2+1)::2,:])
        ds_1_key = model.mlp3(ds_1_corr_feature[:,:,0::2,:])
        
        ds_2_con = model_1.mlp2(ds_2_corr_feature[:,:,1::2,:])
        
        
        ds_1_con = torch.cat(ds_1_con, dim=2)
        ds_2_con = torch.cat(ds_2_con, dim=2)
        ds_1_con = ds_1_con.view(opt.batchsize,-1)
        ds_2_con = ds_2_con.view(opt.batchsize,-1)
        
        
        loss_con_1 = criterion2(ds_1_con.squeeze(), x_cont)
        loss_con_2 = criterion2(ds_2_con.squeeze(), x_cont_1_new)
        
        loss_cat = 0
        for j in range(len(cat_idxs)):
            see1=ds_1_cat[j].squeeze(1)
            see2=x_categ[:,j]
            loss_cat+= criterion1(ds_1_cat[j].squeeze(1),x_categ[:,j])
        
        #Uni-module Contrastive head
        ds_1_feature = (ds_1_feature / ds_1_feature.norm(dim=-1, keepdim=True)).flatten(2,3)
        ds_1_corr_feature = (ds_1_corr_feature / ds_1_corr_feature.norm(dim=-1, keepdim=True)).flatten(2,3)
        
        ds_2_feature = (ds_2_feature / ds_2_feature.norm(dim=-1, keepdim=True)).flatten(2,3)
        ds_2_corr_feature = (ds_2_corr_feature / ds_2_corr_feature.norm(dim=-1, keepdim=True)).flatten(2,3)
        
        
        if opt.pt_projhead_style == 'diff':
            ds_1_feature = model.pt_mlp(ds_1_feature)
            ds_1_corr_feature = model.pt_mlp2(ds_1_corr_feature)
            ds_2_feature = model_1.pt_mlp(ds_2_feature)
            ds_2_corr_feature = model_1.pt_mlp2(ds_2_corr_feature)
            
        elif opt.pt_projhead_style == 'same':
            ds_1_feature = model.pt_mlp(ds_1_feature)
            ds_1_corr_feature = model.pt_mlp(ds_1_corr_feature)
            ds_2_feature = model_1.pt_mlp(ds_2_feature)
            ds_2_corr_feature = model_1.pt_mlp(ds_2_corr_feature)

        ds_1_feature = ds_1_feature.reshape(opt.batchsize, -1)
        ds_2_feature = ds_2_feature.reshape(opt.batchsize, -1)
        
        ds_1_corr_feature = ds_1_corr_feature.reshape(opt.batchsize, -1)
        ds_2_corr_feature = ds_2_corr_feature.reshape(opt.batchsize, -1)
        
        logits_per_aug1 = ds_1_feature @ ds_1_corr_feature.t()/opt.nce_temp
        logits_per_aug1_corr =  ds_1_corr_feature @ ds_1_feature.t()/opt.nce_temp
        targets_1 = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
        loss_1 = criterion1(logits_per_aug1, targets_1)
        loss_1_corr = criterion1(logits_per_aug1_corr, targets_1)
        loss1 = opt.lam0*(loss_1 + loss_1_corr)/2
        
        logits_per_aug2 = ds_2_feature @ ds_2_corr_feature.t()/opt.nce_temp
        logits_per_aug2_corr =  ds_2_corr_feature @ ds_2_feature.t()/opt.nce_temp
        targets_2 = torch.arange(logits_per_aug2.size(0)).to(logits_per_aug2.device)
        loss_2 = criterion1(logits_per_aug2, targets_2)
        loss_2_corr = criterion1(logits_per_aug2_corr, targets_2)
        loss2 = opt.lam0*(loss_2 + loss_2_corr)/2
        
        
        
        print("see")
        
        
        
        





model = pretrain_process_dict(model, trainloader_1, validloader, opt,device, modelsave_path)
