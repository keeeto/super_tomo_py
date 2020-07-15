# -*- coding: utf-8 -*-
""":w
Created on Tue Apr  9 10:19:59 2019

@author: Keith Butler
"""
import os
import numpy as np
import pandas as pd

def normalise_discritise_data(img, mask, flag_multi_class=False, num_class=2, debug=False):
    '''
    Processes the image data to make it ready for the NN. Converts mask into a range of integer values. Normalises the image values.
    
    Args:
        img: The image as an array
        mask: The mask as an array
        flag_multi_class: boolean, is this a multiclass or binary classification
        num_class: integer, the number of classes
    Returns:
        img: The image normalised
        mask: The mask converted to intergers by class
    '''
    if debug:
        print('Maximum value in mask before {0}'.format(mask.max()))

    if(flag_multi_class):
        img = img / 255
        mask = mask[:, :, :, 0] if(len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1]*new_mask.shape[2] \
        ,new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0]* \
        new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    else:
        if(np.max(img) > 1):
            img = img / 255
        if(np.max(mask) > 1):
            mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    if debug:
        print('Maximum value in mask after {0}'.format(mask.max()))
    return (img, mask)

def undersample_dataframe(df, label='label'):
    """
    Balances a class imbalanced dataframe by undersampling. Takes the size of the smallest class
    and cuts all classes to that size, randomly selecting members to fit the proportion.
    
    Args:
        df: The input dataframe (Pandas dataframe object)
	label: The label of the column to sort by (string, default label)
    Returns:
        df_bal: The class balanced dataframe (Pandas dataframe object)

    """
    unique_classes = df[label].unique()
    separate_frames = []
    frame_lengths = []
    for u_class in unique_classes:
        new_frame = df[df[label] == u_class]
        separate_frames.append(new_frame)
        frame_lengths.append(len(new_frame))
    min_number = min(frame_lengths)

    sampled_frames = []
    for frame in separate_frames:
        frac = min_number/len(frame)
        sampled_frames.append(frame.sample(frac=frac))
    return pd.concat(sampled_frames)
    
def build_list_images(datapath, types=None):
    '''
    Builds a list of all image files in a given directory
    
    Args:
        datapath: the location to search
        types: a tuple of the extensions to search, if unspecified a list of defaults is checked
    Returns:
        paths: a list of the datapaths to the images
    '''
    if not types:
        types = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', 
        '.bmp', '.pdf', '.eps', '.ps')
    images = []
    for root, subdirs, files in os.walk(datapath):
        for file in files:
            if os.path.splitext(file)[1].lower() in types:
                images.append(os.path.join(root, file))
    return images
