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

#CT data normalization using three different pixel value ranges 
#Three different min-max normalization ranges were chosen to extract maximal information from the tumor patches 

MIN_BOUND = -1024.0
MAX_BOUND = 100.0
 
def normalize0(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

MIN_BOUND1 = -850.0
MAX_BOUND1 = -400.0  
    
def normalize1(image): 
    image = (image - MIN_BOUND1) / (MAX_BOUND1 - MIN_BOUND1)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

MIN_BOUND2 = -400.0
MAX_BOUND2 = 400.0
    
def normalize2(image):
    image = (image - MIN_BOUND2) / (MAX_BOUND2 - MIN_BOUND2)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

#Then, the normalized CT data were concatenated to form 3 channels

def normcat(image, dimension): #normalization and concatenation for the validation and test sets
    
    numfiles=image.shape[0]
    image=image.reshape(numfiles, dimension, dimension, dimension)
    
    data_images0=[]
    for i in image:
        newimage0=normalize0(i)
        data_images0.append(newimage0)  

    data_images1=[]
    for i in image:
        newimage1=normalize1(i)
        data_images1.append(newimage1)  

    data_images2=[]
    for i in image:
        newimage2=normalize2(i)
        data_images2.append(newimage2)   
        
    data_images0=np.stack(data_images0,axis=0)
    data_images1=np.stack(data_images1,axis=0)
    data_images2=np.stack(data_images2,axis=0)
    
    data_images0=data_images0.reshape(numfiles, dimension, dimension, dimension, 1)
    data_images1=data_images1.reshape(numfiles, dimension, dimension, dimension, 1)
    data_images2=data_images2.reshape(numfiles, dimension, dimension, dimension, 1) 
    
    concatimage=np.concatenate((data_images0, data_images1, data_images2), axis=-1)
    
    return concatimage

def normcat1(image, dimension): #for the data generator (i.e., training set)
     
    data_images0=normalize0(image) 
    data_images1=normalize1(image)
    data_images2=normalize2(image)
         
    data_images0=data_images0.reshape(dimension, dimension, dimension, 1)
    data_images1=data_images1.reshape(dimension, dimension, dimension, 1)
    data_images2=data_images2.reshape(dimension, dimension, dimension, 1) 
    
    concatimage=np.concatenate((data_images0, data_images1, data_images2), axis=-1)
    
    return concatimage 

#Data augmenation

augFlip1 = iaa.Sometimes(1, iaa.Fliplr(1))
augFlip2 = iaa.Sometimes(1, iaa.Flipud(1)) 
augBlur = iaa.Sometimes(1, iaa.GaussianBlur(sigma=(0.0, 1)))
augSharpen = iaa.Sometimes(1, iaa.Sharpen(alpha=0.2, lightness=0.7))
augNoise = iaa.Sometimes(1, iaa.AdditiveGaussianNoise(scale=(0, 100)))

def augvol(image1, image2):
    randomnum = random.randrange(1,16) 
    global imageaug1
    global imageaug2
    if randomnum == 1:
        imageaug1 = ndimage.rotate(image1, axes=(1,0), angle=90, reshape=False)
        imageaug2 = ndimage.rotate(image2, axes=(1,0), angle=90, reshape=False)
    if randomnum == 2:
        imageaug1 = ndimage.rotate(image1, axes=(1,0), angle=180, reshape=False)
        imageaug2 = ndimage.rotate(image2, axes=(1,0), angle=180, reshape=False)
    if randomnum == 3:
        imageaug1 = ndimage.rotate(image1, axes=(1,0), angle=270, reshape=False)
        imageaug2 = ndimage.rotate(image2, axes=(1,0), angle=270, reshape=False)
    if randomnum == 4:
        imageaug1 = ndimage.rotate(image1, axes=(2,0), angle=90, reshape=False) 
        imageaug2 = ndimage.rotate(image2, axes=(2,0), angle=90, reshape=False) 
    if randomnum == 5:
        imageaug1 = ndimage.rotate(image1, axes=(2,0), angle=180, reshape=False)
        imageaug2 = ndimage.rotate(image2, axes=(2,0), angle=180, reshape=False)
    if randomnum == 6:
        imageaug1 = ndimage.rotate(image1, axes=(2,0), angle=270, reshape=False)
        imageaug2 = ndimage.rotate(image2, axes=(2,0), angle=270, reshape=False)
    if randomnum == 7:
        imageaug1 = ndimage.rotate(image1, axes=(2,1), angle=90, reshape=False)
        imageaug2 = ndimage.rotate(image2, axes=(2,1), angle=90, reshape=False)
    if randomnum == 8:
        imageaug1 = ndimage.rotate(image1, axes=(2,1), angle=180, reshape=False)
        imageaug2 = ndimage.rotate(image2, axes=(2,1), angle=180, reshape=False)
    if randomnum == 9:
        imageaug1 = ndimage.rotate(image1, axes=(2,1), angle=270, reshape=False) 
        imageaug2 = ndimage.rotate(image2, axes=(2,1), angle=270, reshape=False)
    if randomnum == 10:
        imageaug1 = np.moveaxis(image1, 0, -1)
        imageaug1 = augFlip1.augment_image(imageaug1)
        imageaug1 = np.moveaxis(imageaug1, 2, 0)  
        
        imageaug2 = np.moveaxis(image2, 0, -1)
        imageaug2 = augFlip1.augment_image(imageaug2)
        imageaug2 = np.moveaxis(imageaug2, 2, 0) 
    if randomnum == 11:
        imageaug1 = np.moveaxis(image1, 0, -1)
        imageaug1 = augFlip2.augment_image(imageaug1)
        imageaug1 = np.moveaxis(imageaug1, 2, 0)  
        
        imageaug2 = np.moveaxis(image2, 0, -1)
        imageaug2 = augFlip2.augment_image(imageaug2)
        imageaug2 = np.moveaxis(imageaug2, 2, 0)  
    if randomnum == 12:
        imageaug1 = augBlur.augment_image(image1) 
        imageaug2 = augBlur.augment_image(image2) 
    if randomnum == 13:
        imageaug1 = augSharpen.augment_image(image1) 
        imageaug2 = augSharpen.augment_image(image2) 
    if randomnum == 14:
        imageaug1 = augNoise.augment_image(image1) 
        imageaug2 = augNoise.augment_image(image2) 
    if randomnum == 15:
        imageaug1 = image1
        imageaug2 = image2
    return imageaug1, imageaug2       

#Data generator (reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
#Two inputs of different dimensions were used: (n, 30, 30, 30, 1) and (n, 60, 60, 60, 1)
#Three-dimensional cubic tumor patch that fit the tumor boundary and eight-times larger cubic patch (twice larger sides) were obtained and used
#The prediction target (labels: either visceral pleural invasion or lymphovascular invasion) is binary

class DataGenerator(keras.utils.Sequence): 
    def __init__(self, list_IDs, inputdata1, inputdata2, labels, augvol, normcat1, batch_size, dim1, dim2, n_channels, shuffle):
        'Initialization'
        self.list_IDs = list_IDs 
        self.inputdata1 = inputdata1 #(n, 30, 30, 30, 1)
        self.inputdata2 = inputdata2 #(n, 60, 60, 60, 1) #eight-times larger patch
        self.labels = labels #prediction target: either visceral pleural invasion or lymphovascular invasion 
        self.batch_size = batch_size
        self.dim1 = dim1
        self.dim2 = dim2
        self.n_channels = n_channels 
        self.augvol=augvol  
        self.normcat1=normcat1   
        self.shuffle = shuffle
        self.on_epoch_end()     
        
    def __len__(self): 
        return int(np.floor(len(self.list_IDs) / self.batch_size))    
    
    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data        
        [X1, X2], y = self.__data_generation(list_IDs_temp)
 
        return [X1, X2], y     
    
    def on_epoch_end(self): 
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
    
    def __data_generation(self, list_IDs_temp): 
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim1, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim2, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image1 = self.inputdata1[ID]
            image1 = image1.reshape(30,30,30)
            
            image2 = self.inputdata2[ID]
            image2 = image2.reshape(60,60,60)
            
            image_aug1, image_aug2 = self.augvol(image1, image2) 
            X1[i,] = self.normcat1(image_aug1, 30) 
            X2[i,] = self.normcat1(image_aug2, 60)    

            # Store class
            y[i] = self.labels[ID]

        return [X1, X2], y   
    
# DenseNet architecture modules

def dense_factor(inputs, filter2, weight_decay, kernel=1, strides=1):  
    x = layers.BatchNormalization()(inputs)
    x = layers.Activation('relu')(x) 
    x = layers.Conv3D(filter2, 
                      kernel, 
                      strides=strides, 
                      kernel_initializer='he_normal', 
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(weight_decay)
                     )(x) 
    return x

def dense_block(x, repetition):  
    for i in range(repetition):
        y = dense_factor(x, 4*filter2, weight_decay)  
        y = dense_factor(y, filter2, weight_decay, 3)
        x = layers.concatenate([y,x], axis=-1)
    return x

def transition_layer(x, compression_factor, droprate, weight_decay): 
    x = layers.BatchNormalization(axis=-1, gamma_regularizer=keras.regularizers.l2(weight_decay), beta_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)
    num_feature_maps = x.shape[-1]
    x = layers.Conv3D( np.floor( compression_factor * num_feature_maps ).astype( np.int ), (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=keras.regularizers.l2(weight_decay))(x)  
    x = layers.AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    return x

def dense_model(filter1, filter2, num_block, compression_factor, droprate, weight_decay): 
    input1 = keras.Input((30, 30, 30, 3))
    input2 = keras.Input((60, 60, 60, 3))
    
    ## 1st input    
    x = layers.BatchNormalization()(input1)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(filter1, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)     
    
    if num_block == 11:
        x = dense_block(x, 1)
        x = transition_layer(x, compression_factor, droprate, weight_decay)
    if num_block == 21:
        for repetition in [2,2]:
            x = dense_block(x, repetition)
            x = transition_layer(x, compression_factor, droprate, weight_decay)
    if num_block == 22:
        for repetition in [2,4]:
            x = dense_block(x, repetition)
            x = transition_layer(x, compression_factor, droprate, weight_decay)
    if num_block == 31:
        for repetition in [2,2,2]:
            x = dense_block(x, repetition)
            x = transition_layer(x, compression_factor, droprate, weight_decay) 
    if num_block == 32:
        for repetition in [2,2,4]:
            x = dense_block(x, repetition)
            x = transition_layer(x, compression_factor, droprate, weight_decay)             
    if num_block == 33:
        for repetition in [2,4,4]:
            x = dense_block(x, repetition)
            x = transition_layer(x, compression_factor, droprate, weight_decay)           
    
    x = layers.GlobalAveragePooling3D()(x)   
    
    ## 2nd input
    x1 = layers.BatchNormalization()(input2)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv3D(filter1, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(weight_decay))(x1)     
    
    if num_block == 11:
        x1 = dense_block(x1, 1)
        x1 = transition_layer(x1, compression_factor, droprate, weight_decay)
    if num_block == 21:
        for repetition in [2,2]:
            x1 = dense_block(x1, repetition)
            x1 = transition_layer(x1, compression_factor, droprate, weight_decay)
    if num_block == 22:
        for repetition in [2,4]:
            x1 = dense_block(x1, repetition)
            x1 = transition_layer(x1, compression_factor, droprate, weight_decay)
    if num_block == 31:
        for repetition in [2,2,2]:
            x1 = dense_block(x1, repetition)
            x1 = transition_layer(x1, compression_factor, droprate, weight_decay) 
    if num_block == 32:
        for repetition in [2,2,4]:
            x1 = dense_block(x1, repetition)
            x1 = transition_layer(x1, compression_factor, droprate, weight_decay)             
    if num_block == 33:
        for repetition in [2,4,4]:
            x1 = dense_block(x1, repetition)
            x1 = transition_layer(x1, compression_factor, droprate, weight_decay)           
    
    x1 = layers.GlobalAveragePooling3D()(x1)   
    x2 = layers.concatenate([x, x1])    
    x3 = layers.Dropout(rate=0.2)(x2) 
    units0= np.floor( x2.shape[-1] /4 ).astype( np.int )
    f1=layers.Dense(units=units0, kernel_initializer='he_normal', bias_initializer= tf.keras.initializers.Constant(0.01))(x3)    
    f2=layers.Dense(units=1, kernel_initializer='zeros', bias_initializer='zeros')(f1)    
    outputs=layers.Activation('sigmoid')(f2)

    model = keras.Model(inputs=[input1, input2], outputs =outputs, name="VPI2_2input_model")
    return model

# Learning rate

cos_decay_ann = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=0.001, first_decay_steps=60, t_mul=2, m_mul=0.95, alpha=0.01) 

# Model with exampleary hyperparameters

no_epoch=800
params = {'dim1':(30,30,30),
          'dim2':(60,60,60), 
          'batch_size': 20,  
          'n_channels': 3,
          'shuffle':True
         }

droprate= 0.2 
weight_decay=1E-4 
filter1=64    
filter2=32
num_block=32
compression_factor=0.7

model = dense_model(filter1, filter2, num_block, compression_factor, droprate, weight_decay)   
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=cos_decay_ann), metrics=["AUC"])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40) 
csv_logger = keras.callbacks.CSVLogger('model log.csv', append=False, separator=';')
checkpointer = keras.callbacks.ModelCheckpoint(filepath='bestmodel.h5', verbose=1, save_best_only=True, monitor='val_auc', mode='max')
history=model.fit_generator(DataGenerator(train_ID, train_image1F, train_image2F, train_VPI, 
                                          augvol, normcat1, **params),
                            epochs=no_epoch,
                            validation_data=([tune_image1FNC, tune_image2FNC], tune_VPI),
                            verbose=1, 
                            callbacks=[early_stopping, csv_logger, checkpointer])
model.save('model.h5')
