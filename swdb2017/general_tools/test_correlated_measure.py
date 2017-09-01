## test correlated matrix 

import numpy as np
import pandas as pd
import correlated_measure as cc

d = pd.read_csv("/Users/jewellsean/Downloads/Faces.csv")
kkey = 'Faces'

def labell(name): 
    if (name == 'Faces'):
        return 1
    else:
        return 0

d['val'] = d['Faces'].apply(labell)
model_mtx = np.outer(np.array(d['val']), np.array(d['val']) )
permuted_ind = np.array(d.sort_values('Faces')['Image']) - 1

emp_mtx = np.zeros_like(model_mtx)
n_samples = 100

out = cc.calculate_sampling_dist(model_mtx, emp_mtx, n_samples)

