from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import  pandas as pd
pd.options.mode.chained_assignment = None
def Pre_Processing(x,y,selectedClass=0,text='train'):
    Columns = x.columns
    for i in Columns:
        try:
            found = x[i].str.contains('[a-zA-Z]', regex=True)
            if found.sum() > 0:
                x[i].replace(['female', 'male'], [0, 1], inplace=True)
                x[i] = x[i].replace(np.NAN, x[i].value_counts().idxmax())

        except:
            x[i] = x[i].fillna(value=0)
            x[i] = ((x[i] - x[i].mean()) / (x[i].std()))
    if text=='train':
        y.replace(selectedClass,[0,1,-1],inplace=True)
    return x,y