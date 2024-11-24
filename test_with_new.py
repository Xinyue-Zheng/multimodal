import torch
from torch import nn
from models import SAINT, SAINT_vision,SAINT_con

import pandas as pd
import torch.nn.functional as F

from data_openml import DataSetCatCon, data_prep_df,DataSetCatCon_without_target
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
from pretraining import SAINT_pretrain,pretrain_process
from with_cons import ReconstructionHead, SAINT_Reconstruction, mask_with_0, cal_correct
import pickle as pkl

import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', required=True, type=int)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', required=True, type=str,choices = ['binary','multiclass','regression'])
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
parser.add_argument('--pretrain_epochs', default=1, type=int)
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
cat_columns=['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16','Class']
cont_columns=['V6', 'V13', 'V1', 'V12', 'V15', 'V14', 'V10']

cat_dims, cat_idxs, con_idxs, X_train, X_valid, X_test, train_mean, train_std = data_prep_df(df,cat_columns,cont_columns, datasplit=[.65, .15, .2])
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

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

train_ds = DataSetCatCon_without_target(X_train, cat_idxs, continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon_without_target(X_valid, cat_idxs, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon_without_target(X_test, cat_idxs, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

model = SAINT_con(
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
model = pretrain_process(model, trainloader, validloader, opt,device, modelsave_path)

for param in model.parameters():
    param.requires_grad = False

reconstruction_head=ReconstructionHead(
                    d_in=opt.embedding_size,
                    bias=True,
                    activation= "relu",
                    normalization="layer_norm",
                    n_num_features=len(cont_columns),
                    category_sizes=cat_dims,
                )


model_with_head = SAINT_Reconstruction(model, reconstruction_head)
model_with_head.to(device)
vision_dset = opt.vision_dset
optimizer = optim.AdamW(model.parameters(),lr=0.0001)
threshold_acc = 0.5
model_save_path_acc = "model_acc"
accuracy_list = []
for epoch in range(opt.epochs):
    model_with_head.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
        x_categ_c, c_cat = mask_with_0(x_categ, 0.2)
        x_cont_c, c_con = mask_with_0(x_cont, 0.2)
        
        x_categ_c=x_categ_c.long()
        # test_see_1 = torch.nonzero(c_cat)
        # a=test_see_1[:, 0]
        # b=test_see_1[:, 1]
        _ , x_categ_enc_c, x_cont_enc_c = embed_data_mask(x_categ_c, x_cont_c, cat_mask, con_mask,model,vision_dset)
        
        x_categ_enc_c.to(device)
        x_cont_enc_c.to(device)
        output=model_with_head(x_categ_enc_c, x_cont_enc_c)
        
        loss1=0
        for i in range(len(output["cat_out"])):
            pr=output["cat_out"][i]
            gr=x_categ[:,i]
            loss1 += F.cross_entropy(pr, gr.long())
        
        # correct=0
        # total = 0
        # for idxes in test_see_1:
        #     list_idx = idxes[1]
        #     tensor_idx = idxes[0]
        #     sp_out_cat=output["cat_out"][list_idx]
        #     sp_out_cat=sp_out_cat[tensor_idx,:]
        #     pre_result = torch.argmax(sp_out_cat).item()
        #     ground_truth = x_categ[idxes[0], idxes[1]].item()
        #     if pre_result == ground_truth:
        #         correct=correct+1
        #         total = total+1
        #     else: 
        #         total = total+1
            
        
        loss2 = F.mse_loss(output["num_out"], x_cont)
        loss = loss1+loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if epoch%5==0:
        total_number = 0
        correct_number = 0 
        for i, data_eval in enumerate(validloader):
            model_with_head.eval()
            model.eval()
            with torch.no_grad():
                correct_num, total_num = cal_correct(model_with_head,model,opt, data, device)
                correct_number = correct_number+correct_num
                total_number = total_number+total_num
                # x_categ, x_cont,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
                # x_categ_c, c_cat = mask_with_0(x_categ, 0.2)
                # x_cont_c, c_con = mask_with_0(x_cont, 0.2)
                # _ , x_categ_enc_c, x_cont_enc_c = embed_data_mask(x_categ_c, x_cont_c, cat_mask, con_mask,model,vision_dset)
                # x_categ_enc_c.to(device)
                # x_cont_enc_c.to(device)
                # output=model_with_head(x_categ_enc_c, x_cont_enc_c)
                # cat = output["cat_out"]
                
                # c_indecies = torch.nonzero(c_cat)
                
                
                # correct=0
                # total = 0
                # for idxes in c_indecies:
                #     list_idx = idxes[1]
                #     tensor_idx = idxes[0]
                #     sp_out_cat=output["cat_out"][list_idx]
                #     sp_out_cat=sp_out_cat[tensor_idx,:]
                #     pre_result = torch.argmax(sp_out_cat).item()
                #     ground_truth = x_categ[idxes[0], idxes[1]].item()
                #     if pre_result == ground_truth:
                #         correct=correct+1
                #         total = total+1
                #     else: 
                #         total = total+1
        
        accuracy = correct_number/total_number
        accuracy_list.append(accuracy)
        if accuracy>threshold_acc:
            torch.save(model.state_dict(),f'{model_save_path_acc}/{epoch}_{accuracy}.pth')
            print(f'model with cat acc: {accuracy} saved ')       
        print('eval_1')

with open('acc_list.pkl', 'wb') as f:
    pkl.dump(accuracy_list, f)           
                
                
            
        

