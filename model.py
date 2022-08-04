import openpyxl
import os
import numpy as np
import scipy
from scipy import ndimage   
from sklearn.model_selection import train_test_split
import pandas as pd 
import random 
from imgaug import augmenters as iaa
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras import layers 

#CT data normalization 

MIN_BOUND = -1024.0
MAX_BOUND = 100.0
    
def normalize0(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

#
MIN_BOUND1 = -850.0
MAX_BOUND1 = -400.0 #ref for -400 HU: PLoS One. 2018; 13(10): e0205490.
    
def normalize1(image): #may not be useful!
    image = (image - MIN_BOUND1) / (MAX_BOUND1 - MIN_BOUND1)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

#
MIN_BOUND2 = -400.0
MAX_BOUND2 = 400.0
    
def normalize2(image):
    image = (image - MIN_BOUND2) / (MAX_BOUND2 - MIN_BOUND2)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 
