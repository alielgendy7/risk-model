# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 01:57:25 2022

@author: a
"""

import pickle
import pandas as pd

# load pickle model
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)

def predict_level(data_dict):
    # convert input dictionary into data frame
    df = pd.DataFrame([data_dict])

    # make predictions
    prediction = model.predict(df)

    return prediction