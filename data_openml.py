import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d

def data_split_df(X,nan_mask,indices, column_key_embeddings):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices],
        'key_embed': np.tile(column_key_embeddings, (X.values[indices].shape[0],1))
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'

    return x_d

def transform_timestamps_to_input(timestamps):
    """
    Transforms a datetime64[ns] array into a numpy array with two columns:
    - Day of the year (normalized between 0 and 1).
    - Time of day (normalized between 0 and 1).
    
    Args:
        timestamps (np.ndarray): Numpy array of timestamps in datetime64[ns] format.
    
    Returns:
        np.ndarray: Array of shape (len(timestamps), 2) with normalized day of year and time of day.
    """
    # Convert to pandas datetime for easy extraction of components
    timestamps = pd.to_datetime(timestamps)

    # Normalize day of the year (0 to 1)
    day_of_year = timestamps.dayofyear / 365.0

    # Normalize time of day (0 to 1)
    time_of_day = (timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600) / 24.0

    # Stack day_of_year and time_of_day into a 2D array
    t_input = np.stack([day_of_year, time_of_day], axis=1)
    
    return t_input

def data_split_df_time(X,nan_mask,indices, column_key_embeddings, time):
    time_stamps = transform_timestamps_to_input(np.array(X[time][indices]))
    
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices],
        'key_embeds': np.tile(column_key_embeddings, (X.values[indices].shape[0],1)),
        'time_stamp': time_stamps
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'

    return x_d

def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):
    
    np.random.seed(seed) 
    dataset = openml.datasets.get_dataset(ds_id)
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    if ds_id == 42178:
        categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp ]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728,42705,42729,42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idx_test=sorted([X.columns.get_loc(col) for col in categorical_columns])
    con_idx_test=sorted([X.columns.get_loc(col) for col in cont_columns])

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    
    
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std

def data_prep_keyvalue(data_dict, categorical_columns, cont_columns, datasplit=[.65, .15, .2]):
    # Step 1: Embed the keys as categorical features
    all_keys = list(data_dict[0].keys())  # Assuming all dictionaries have the same keys
    key_encoder = LabelEncoder()
    key_encoder.fit(all_keys)
    key_dims = len(key_encoder.classes_)
    
    # Assign an index to each key based on the embedding
    for entry in data_dict:
        entry["key_emb"] = [key_encoder.transform([k])[0] for k in entry.keys()]

    # Add a "Set" key to each dictionary entry to split data into train, valid, and test sets
    for entry in data_dict:
        entry["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit)
    
    train_data = [entry for entry in data_dict if entry["Set"] == "train"]
    valid_data = [entry for entry in data_dict if entry["Set"] == "valid"]
    test_data = [entry for entry in data_dict if entry["Set"] == "test"]
    # Remove the "Set" key after assigning splits
    # for entry in data_dict:
    #     entry.pop("Set", None)
    
    # Initialize missing value mask
    nan_mask = []

    # Handle categorical columns: Track missing values, then apply Label Encoding
    cat_dims = []
    for col in categorical_columns:
        # Track missing values before encoding
        col_nan_mask = [(0 if entry[col] is None else 1) for entry in data_dict]
        nan_mask.append(col_nan_mask)
        
        # Replace None with "MissingValue" and apply Label Encoding
        unique_values = set(entry[col] for entry in data_dict if entry[col] is not None)
        l_enc = LabelEncoder()
        l_enc.fit(list(unique_values) + ["MissingValue"])  # Include "MissingValue" for NaNs
        cat_dims.append(len(l_enc.classes_))
        
        for entry in data_dict:
            if entry[col] != entry[col]:
                entry[col] = "MissingValue"
            entry[col] = l_enc.transform([entry[col]])[0]

    # Handle continuous columns: Fill missing values with the mean of the train split and add mask
    for col in cont_columns:
        # Calculate mean from train data
        train_values = [entry[col] for entry in train_data if entry[col] is not None]
        train_mean = np.mean(train_values) if train_values else 0
        
        # Create nan mask and fill missing values
        col_nan_mask = [(1 if entry[col] is not None else 0) for entry in data_dict]
        nan_mask.append(col_nan_mask)
        
        for entry in data_dict:
            if entry[col] is None:
                entry[col] = train_mean  # Fill missing values for continuous columns

    # Prepare final nan_mask array by transposing rows and columns
    nan_mask = np.array(nan_mask).T  # shape: [num_entries, num_columns]

    # Split data and prepare for output
    def data_split_dict(data, cont_columns, cat_dims, key_dims, nan_mask):
        data_list = []
        key_embeds = []
        
        for entry in data:
            row_data = []
            row_key_emb = entry["key_emb"]
            
            # Process each column's value
            for col in categorical_columns + cont_columns:
                row_data.append(entry[col])
                
            data_list.append(row_data)
            key_embeds.append(row_key_emb)
        
        return {
            "data": np.array(data_list),
            "nan_mask": nan_mask,
            "key_embeds": np.array(key_embeds)
        }

    # Splitting the nan_mask by data split
    nan_mask_train = nan_mask[[i for i, entry in enumerate(data_dict) if entry["Set"] == "train"]]
    nan_mask_valid = nan_mask[[i for i, entry in enumerate(data_dict) if entry["Set"] == "valid"]]
    nan_mask_test = nan_mask[[i for i, entry in enumerate(data_dict) if entry["Set"] == "test"]]
    for entry in data_dict:
         entry.pop("Set", None)
    X_train = data_split_dict(train_data, cont_columns, cat_dims, key_dims, nan_mask_train)
    X_valid = data_split_dict(valid_data, cont_columns, cat_dims, key_dims, nan_mask_valid)
    X_test = data_split_dict(test_data, cont_columns, cat_dims, key_dims, nan_mask_test)
    
    # Calculate train mean and standard deviation for continuous columns
    train_cont_data = np.array([entry[col] for entry in train_data for col in cont_columns], dtype=np.float32)
    train_mean, train_std = train_cont_data.mean(axis=0), train_cont_data.std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    
    cat_idxs = list(range(len(categorical_columns)))
    con_idxs = list(range(len(categorical_columns), len(categorical_columns) + len(cont_columns)))
    
    return cat_dims, cat_idxs, con_idxs, key_dims, X_train, X_valid, X_test, train_mean, train_std




def data_prep_df(X, categorical_columns, cont_columns,keys, time_step_head, datasplit=[.65, .15, .2]):
    key_encoder = LabelEncoder()
    encoded_keys = key_encoder.fit_transform(keys)  # Encode each column header to an integer ID
    key_dims = len(key_encoder.classes_)  # Number of unique keys
    
    # Create a dictionary to store the column key embeddings
    #column_key_embeddings = {col: enc_key for col, enc_key in zip(keys, encoded_keys)}
    column_key_embeddings = np.array([encoded_keys[X.columns.get_loc(col)] for col in X.columns if col is not time_step_head])

    cat_idxs = []
    con_idxs = []
    if categorical_columns is not None: 
        for col in categorical_columns:
            X[col] = X[col].astype("str")
        cat_idxs=sorted([X.columns.get_loc(col) for col in categorical_columns])
    
    if cont_columns is not None:
        con_idxs=sorted([X.columns.get_loc(col) for col in cont_columns])

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    if categorical_columns is not None: 
        
        for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
            X[col] = X[col].fillna("MissingValue")
            l_enc = LabelEncoder() 
            X[col] = l_enc.fit_transform(X[col].values)
            cat_dims.append(len(l_enc.classes_))
    
    
    if cont_columns is not None:
        for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)
        
    #see = np.array(X[time_step_head][train_indices])
    X_train = data_split_df_time(X,nan_mask,train_indices, column_key_embeddings,time_step_head)
    X_valid = data_split_df_time(X,nan_mask,valid_indices, column_key_embeddings,time_step_head)
    X_test = data_split_df_time(X,nan_mask,test_indices, column_key_embeddings,time_step_head)
    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    
    return cat_dims, cat_idxs, con_idxs,key_dims, X_train, X_valid, X_test, train_mean, train_std


class DataSetCatCon_dict(Dataset):
    def __init__(self, X, cat_cols,time_idx, continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        self.key = X['key_embeds'].copy().astype(np.int64)
        self.time_stamp = X['time_stamp'].copy().astype(np.float32)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols) - set([time_idx]))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        # self.X_key = key[:,:].copy().astype(np.int64)
        # self.cls = np.zeros_like(self.y,dtype=int)
        # self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        length=self.X2.shape[0]
        return length
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return self.X1[idx], self.X2[idx], self.X1_mask[idx], self.X2_mask[idx], self.key[idx], self.time_stamp[idx]

class DataSetCatCon_align(Dataset):
    def __init__(self, X, X_N, cat_cols, cat_cols_N,time_idx, time_idx_N, continuous_mean_std=None, continuous_mean_std_N=None):
        
        cat_cols = list(cat_cols)
        self.key = X['key_embeds'].copy().astype(np.int64)
        self.time_stamp = X['time_stamp'].copy().astype(np.float32)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols) - set([time_idx]))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        # self.X_key = key[:,:].copy().astype(np.int64)
        # self.cls = np.zeros_like(self.y,dtype=int)
        # self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
            
            
        cat_cols_N = list(cat_cols_N)
        self.key_N = X_N['key_embeds'].copy().astype(np.int64)
        self.time_stamp_N = X_N['time_stamp'].copy().astype(np.float32)
        X_mask_N =  X_N['mask'].copy()
        X_N = X_N['data'].copy()
        
        con_cols_N = list(set(np.arange(X_N.shape[1])) - set(cat_cols_N) - set([time_idx_N]))
        self.X1_N = X_N[:,cat_cols_N].copy().astype(np.int64) #categorical columns
        self.X2_N = X_N[:,con_cols_N].copy().astype(np.float32) #numerical columns
        self.X1_mask_N = X_mask_N[:,cat_cols_N].copy().astype(np.int64) #categorical columns
        self.X2_mask_N = X_mask_N[:,con_cols_N].copy().astype(np.int64) #numerical columns
        
        # self.X_key = key[:,:].copy().astype(np.int64)
        # self.cls = np.zeros_like(self.y,dtype=int)
        # self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std_N is not None:
            mean_N, std_N = continuous_mean_std_N
            self.X2_N = (self.X2_N - mean_N) / std_N

    def __len__(self):
        length=self.X2.shape[0]
        return length
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return self.X1[idx], self.X2[idx], self.X1_mask[idx], self.X2_mask[idx], self.key[idx], self.X1_N[idx], self.X2_N[idx], self.X1_mask_N[idx], self.X2_mask_N[idx], self.key_N[idx], self.time_stamp[idx], 




class DataSetCatCon_without_target(Dataset):
    def __init__(self, X, cat_cols,continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        # self.cls = np.zeros_like(self.y,dtype=int)
        # self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        length=self.X2.shape[0]
        return length
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return self.X1[idx], self.X2[idx], self.X1_mask[idx], self.X2_mask[idx]



class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
            

    def __len__(self):
        print(len(self.y))
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

