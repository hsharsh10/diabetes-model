# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:42:15 2025

@author: Harsh
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/Harsh/Desktop/projects/trained_model.sav','rb'))

def diabetes_prediction(input_data):
    input_data =(5,166,72,19,175,25.8,0.587,51)

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction =loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] ==0):
        print('not diabetic')
    else:
        print('diabetic')
    
    
def main()


