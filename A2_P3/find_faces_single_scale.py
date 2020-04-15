"""
 Princeton University, COS 429, Fall 2019
"""
import numpy as np
from hog36 import hog36
import math 
import os
import cv2
import random
from glob import glob
from logistic_prob import logistic_prob

def find_faces_single_scale(img, stride, thresh, params, orientations, wrap180):
    """Find 36x36 faces in an image

    Args:
        img: an image
        stride: how far to move between locations at which the detector is run
        thresh: probability threshold for calling a detection a face
        params: trained face classifier parameters
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360

    Returns:
        outimg: copy of img with face locations marked
        probmap: probability map of face detections
    """
    windowsize = 36
    if stride > windowsize:
        stride = windowsize

    height, width = img.shape
    probmap = np.zeros([height, width])
    outimg = np.array(img)

    # Loop over windowsize x windowsize windows, advancing by stride
    hog_descriptor_size = 100 * orientations
    window_descriptor = np.zeros([hog_descriptor_size + 1, 1])

    # // slides down then across the image, by stride
    for i in range(0, width-windowsize, stride):
        for j in range(0, height-windowsize, stride):

            # Crop out a windowsize x windowsize window starting at (i,j)
            crop = img[i:i+windowsize,j:j+windowsize] 

            # Compute a HoG descriptor, and run the classifier
            window_descriptor[0,0] = 1
            window_descriptor[1:,0] = hog36(crop, orientations, wrap180)
            # NEED TO TRAIN AND RUN CLASSIFIER ?? PROB --> FIT () ?? or since trained params good 
            probability = logistic_prob(window_descriptor, params) #or need to do both fit + prob 

            # Mark detection probability in probmap
            win_i = i + int((windowsize - stride) / 2)
            win_j = j + int((windowsize - stride) / 2)
            probmap[win_i:win_i+stride, win_j:win_j+stride] = probability

            # If probability of a face is below thresh, continue 
            # else mark the face on img 
            if probability < thresh:
                continue
             
            #print("got here")
            # Mark the face in outimg
            outimg[i, j:j+windowsize] = 255
            outimg[i+windowsize-1, j:j+windowsize] = 255
            outimg[i:i+windowsize, j] = 255
            outimg[i:i+windowsize, j+windowsize-1] = 255

    return outimg, probmap
