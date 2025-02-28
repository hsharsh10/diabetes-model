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
    
    
def main():
 st.title('diabetes prediction')
    
    pregnancies = st.text_input('no. of pregnancies')
    glucose = st.text_input('glucose level')
    bloodpressure = st.text_input('bp value')
    skinthickness = st.text_input('skin thickness value')
    insulin = st.text_input('insulin value')
    bmi = st.text_input('bmi value')
    age = st.text_input('age of a person')
    diabtetespedigreefunction = st.text_input('diabtetes pedigree function value')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,age,diabtetespedigreefunction])

    st.success(diagnosis) 
    
    
if __name__ == '__main__':
    main()


