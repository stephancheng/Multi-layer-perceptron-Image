# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:39:08 2021

@author: Stephan
reference: https://github.com/87surendra/Random-Forest-Image-Classification-using-Python/blob/master/Random-Forest-Image-Classification-using-Python.py
The folling code will read image, turn into feature then save in csv format
"""

# Importing all the necessary libraries
from sklearn.preprocessing import MinMaxScaler
import mahotas
import cv2
import numpy as np
import pandas as pd
 

# features description -1:  Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor -2 Haralick Texture 
def fd_haralick(image):
    # conver the image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Ccompute the haralick texture fetature ve tor 
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic

# feature-description -3 Color Histogram
def fd_histogram(image, bins, mask=None):
    # conver the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #COPUTE THE COLOR HISTPGRAM
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist,hist)
    # return the histog....
    return hist[0:6].flatten() # only 0 to 6 bins are inclued because the remaining are 0

def readdata(path):
    df = pd.read_csv(path, sep= ' ', header=None, names=['img_link', 'class'])
    return df

def create_feature(img_link, fixed_size, bins):
        image = cv2.imread(img_link) 
        if image is not None:
            image = cv2.resize(image,fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image, bins)
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            return global_feature

def main():  
    # bins for histograms 
    bins = 8
    # make a fix file size
    fixed_size  = tuple((500,500))    
    # make a scaler
    scaler = MinMaxScaler(feature_range=(0, 1))    
    # read the data reference file
    train_ref, val_ref, test_ref = readdata("train.txt"), readdata("val.txt"), readdata("test.txt")      
    
    '''
    In each link from the data reference file,
    use the create_feature function to open the image,
    3 csv files are created with training, validation and testing features
    The order will remain the same as the data refernce file
    '''    
    x_train = pd.DataFrame([create_feature(i, fixed_size, bins) for i in train_ref['img_link']])
    x_train = scaler.fit_transform(x_train)
    print("finish loaded training image")
    x_val = pd.DataFrame([create_feature(i, fixed_size, bins) for i in val_ref['img_link']])
    x_val = scaler.fit_transform(x_val)
    print("finish loaded validation image")
    x_test = pd.DataFrame([create_feature(i, fixed_size, bins) for i in test_ref['img_link']])
    x_test = scaler.fit_transform(x_test)
    print("finish loaded test image")    
    
    x_train,x_val, x_test = pd.DataFrame(x_train), pd.DataFrame(x_val), pd.DataFrame(x_test)
    
    x_train.to_csv('x_train.csv')
    x_val.to_csv('x_val.csv')
    x_test.to_csv('x_test.csv')

if __name__ == "__main__":
    main()