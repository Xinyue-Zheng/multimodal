import pickle as pkl
import pandas as pd


# see = pd.read_csv('model.csv')
# print(see)
with open ('model.pkl', 'rb') as f:
    model_data = pkl.load(f)
    see = model_data['biomarkers']
    see2 = model_data['tide']
    p = pd.DataFrame(see)
    p.to_csv('biomarkers.csv')
    print(model_data)