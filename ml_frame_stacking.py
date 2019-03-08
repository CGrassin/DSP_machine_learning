#!/usr/bin/env python3
import cv2
import os
from scipy.signal import fftconvolve,gaussian
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2DTranspose
import matplotlib.pylab as plt

# This code is a proof of concept for machine learning 
# astronomy picture deconvolution. It is NOT a finished
# application.

# Path of the training & test set
train_path = "../Astronomy_all/"
test_path = "../Astronomy_test_data/"

def plti(im, grey=1, path=None):
    """
    Helper function to plot an image.
    """
    plt.figure()
    if grey :
        plt.gray()
        im = im.reshape(len(im), len(im[0]))
    plt.imshow(im)
    plt.axis('off')
    if path != None:
        plt.imsave(path, im, format="png", cmap="gray")

def rgb2gray(rgb):
    return (np.dot(rgb[...,:3], [0.299, 0.587, 0.114]))

def corrupt_picture(im,
                    conv_window=10,
                    y_blur=0,
                    x_blur=0,
                    noise_factor=0):
    # Convolution (glow/blur)
    window = np.outer(gaussian(conv_window, std=x_blur),
                      gaussian(conv_window, std=y_blur)
                      )
    window /= np.sum(window)
    im = fftconvolve(im,window, mode='same')
    
    # Add noise
    im = im + noise_factor * np.random.randn(im.shape[0],im.shape[1])
    
    # Delete line of pixels
#    im[np.random.randint(0,len(im)),:] = 0
    
    # Clip
    im = np.clip(im, 0., 1.)
    
    return im

def open_all_pictures(folderpath):
    pictures = []
    print("Opening files in " + folderpath)
    directory = os.fsencode(folderpath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print("Opening " + filename)
        
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            pic = cv2.imread(os.path.join(folderpath, filename))
            pic = pic.astype("float32") / 255
            pictures.append(pic)
    return pictures

def generate_corrupted_frames(pict,nb_frames,y_blur,x_blur,noise_factor):
    picture_cor = np.empty([len(pict), len(pict[0]),nb_frames])
    for i in range(nb_frames):
        picture_cor[:,:,i] = corrupt_picture(pict,
                   y_blur=y_blur,
                   x_blur=x_blur,
                   noise_factor=noise_factor)
    return picture_cor

def generate_sets(path, nb_stack):
    pictures = open_all_pictures(path)
    corrupted_pictures = []

    # Generate corrupted input data set
    for idx, picture in enumerate(pictures):
        picture =  rgb2gray(picture)
        picture = cv2.resize(picture,dsize=(160, 120))
        
        # Generate 'nb_pic' corrupted frames for each picture    
        corrupted_pictures.append(
                generate_corrupted_frames(picture,
                                          nb_stack,
                                          0.01+pow(np.random.ranf(),1/4),
                                          0.01+pow(np.random.ranf(),1/4),
                                          0.05))
        
        picture = np.reshape(picture,(len(picture), len(picture[0]),1))
        pictures[idx]=picture
    
    return pictures,corrupted_pictures

# Open pictures
nb_pic = 8
pictures,corrupted_pictures = generate_sets(train_path,nb_pic)

# Model (https://blog.keras.io/building-autoencoders-in-keras.html)
input_img = Input(shape=(len(pictures[0]), len(pictures[0][0]),nb_pic))
x = Dense(nb_pic, activation='linear')(input_img)
x = Dense(nb_pic*2, activation='linear')(x)
x = Dense(nb_pic*2, activation='linear')(x)
x = Dense(nb_pic*2, activation='linear')(x)
x = Conv2DTranspose(nb_pic*4, (2, 2), activation='relu', padding='same')(x)
x = Dense(nb_pic*2, activation='linear')(x)
x = Conv2DTranspose(nb_pic*4, (2, 2), activation='relu', padding='same')(x)
x = Dense(nb_pic*2, activation='linear')(x)
x = Dense(nb_pic*2, activation='linear')(x)
x = Dense(nb_pic*2, activation='linear')(x)
x = Dense(nb_pic*2, activation='linear')(x)
decoded = Dense(1, activation='relu')(x)
model = Model(input_img, decoded)

model.compile(optimizer='Nadam',loss='mse')
model.fit(np.array(corrupted_pictures), 
                np.array(pictures),
                epochs=50,
                batch_size=10,
                shuffle=True)

## Predict example
pictures,corrupted_pictures = generate_sets(test_path,nb_pic)
for idx in range(len(pictures)) :
    picture = pictures[idx]
    plti(picture,grey=1)
    picture = corrupted_pictures[idx]
    plti(picture[:,:,0],grey=1)
    picture = model.predict(np.array([picture]), batch_size=128)[0]
    plti(picture,grey=1)