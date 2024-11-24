import torch
import numpy as np
import torch.nn as nn

class FixedTime2VecMultiCycle(torch.nn.Module):
    def __init__(self, device, time_dim=12):
        """
        Initializes the Fixed Time2Vec layer with multiple preset cycles.
        Args:
            time_dim (int): Total number of dimensions in the time embedding. 
                            It should be a multiple of the number of cycles to ensure even distribution.
        """
        super(FixedTime2VecMultiCycle, self).__init__()
        
        # Set fixed frequencies for different cycles (e.g., daily, weekly, yearly)
        base_frequencies = torch.tensor([
            2 * np.pi,         # Daily cycle
            2 * np.pi / 365    # Yearly cycle
        ], dtype=torch.float32)
        
        # Repeat these frequencies to match the specified time_dim
        num_cycles = len(base_frequencies)
        assert time_dim % num_cycles == 0, "time_dim should be a multiple of the number of cycles."
        repeats_per_cycle = time_dim // num_cycles
        self.frequencies = base_frequencies.repeat_interleave(repeats_per_cycle).to(device)
        
        # Create biases for each frequency (all set to zero for fixed version)
        self.biases = torch.zeros(time_dim, dtype=torch.float32).to(device)
    
    def forward(self, t):
        """
        Forward pass for Fixed Time2Vec with multiple cycles.
        Args:
            t (torch.Tensor): Input tensor with columns for each time component (e.g., day of year, time of day).
                              Shape should be (batch_size, num_time_components).
        Returns:
            torch.Tensor: Time embeddings of shape (batch_size, time_dim).
        """
        # Expand t to match the size of self.frequencies
        expanded_t = t.repeat(1, len(self.frequencies) // t.shape[1])
        
        # Apply sine transformations with preset frequencies
        time_embedding = torch.sin(self.frequencies * expanded_t + self.biases)
        
        return time_embedding


def embed_data_dict(x_categ, x_cont, cat_mask, con_mask,model,key,time_stamp,vision_dset=False):
    
    device = x_cont.device
    time2vec_layer = FixedTime2VecMultiCycle(device,time_dim=16).to(device)
    time_emb = time2vec_layer(time_stamp)
    key_emb = model.key_embeds(key)
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    


    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)


    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]
    
    cat_result = torch.empty(x_categ_enc.shape[0], 2 * x_categ_enc.shape[1], x_categ_enc.shape[2]).to(device)
    cat_result[:, 1::2, :] = x_categ_enc  # Assign every other "layer" with tensor1
    cat_result[:, 0::2, :] = key_emb[:, :x_categ_enc.shape[1],:]
    
    cont_result = torch.empty(x_cont_enc.shape[0], 2 * x_cont_enc.shape[1], x_cont_enc.shape[2]).to(device)
    cont_result[:, 1::2, :] = x_cont_enc  # Assign every other "layer" with tensor1
    cont_result[:, 0::2, :] = key_emb[:, :x_cont_enc.shape[1],:]
    
    cat_token_encoding = torch.tensor([1, 0]).repeat( x_categ_enc.shape[1]+ 1)[:2 * x_categ_enc.shape[1]]
    cat_token_encoding = cat_token_encoding.repeat(cat_result.shape[0], 1).to(device)
    
    cont_token_encoding = torch.tensor([1, 0]).repeat( x_cont_enc.shape[1]+ 1)[:2 * x_cont_enc.shape[1]]
    cont_token_encoding = cont_token_encoding.repeat(cont_result.shape[0], 1).to(device)
    
    cat_token_embedding = model.token_type_embedding(cat_token_encoding)
    con_token_embedding = model.token_type_embedding(cont_token_encoding)
    
    cat_result = cat_result+cat_token_embedding+time_emb.unsqueeze(1).repeat(1, cat_result.shape[1], 1)
    cont_result = cont_result+con_token_embedding +time_emb.unsqueeze(1).repeat(1, cont_result.shape[1], 1)
    if vision_dset:
        
        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(device)
        pos_enc =model.pos_encodings(pos)
        x_categ_enc+=pos_enc

    return x_categ, cat_result, cont_result


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    


    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)


    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        
        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(device)
        pos_enc =model.pos_encodings(pos)
        x_categ_enc+=pos_enc

    return x_categ, x_categ_enc, x_cont_enc




def mixup_data(x1, x2 , lam=1.0, y= None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)


    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    return mixed_x1, mixed_x2


def add_noise(x_categ,x_cont, noise_params = {'noise_type' : ['cutmix'],'lambda' : 0.1}):
    lam = noise_params['lambda']
    device = x_categ.device
    batch_size = x_categ.size()[0]

    if 'cutmix' in noise_params['noise_type']:
        index = torch.randperm(batch_size)
        cat_corr = torch.from_numpy(np.random.choice(2,(x_categ.shape),p=[lam,1-lam])).to(device)
        con_corr = torch.from_numpy(np.random.choice(2,(x_cont.shape),p=[lam,1-lam])).to(device)
        x1, x2 =  x_categ[index,:], x_cont[index,:]
        x_categ_corr, x_cont_corr = x_categ.clone().detach() ,x_cont.clone().detach()
        x_categ_corr[cat_corr==0] = x1[cat_corr==0]
        x_cont_corr[con_corr==0] = x2[con_corr==0]
        return x_categ_corr, x_cont_corr
    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2,(x_categ.shape),p=[lam,1-lam])
        x_cont_mask = np.random.choice(2,(x_cont.shape),p=[lam,1-lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ,x_categ_mask), torch.mul(x_cont,x_cont_mask)
        
    else:
        print("yet to write this")
