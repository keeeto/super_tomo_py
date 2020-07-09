import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import PIL

def _find_image_files(datapath, ftypes):
    '''
    In a a given datapath return all files of types listed in ftypes
    Args:
        datapath: (str) the location of the original image(s).
        ftypes: (list) the filetypes to look for eg ['.tif'].
    '''
    images = []
    for root, subdirs, files in os.walk(datapath):
        for file in files:
            if os.path.splitext(file)[1].lower() in ftypes:
                images.append(os.path.join(root, file))
    return images

def image_to_windows(datapath, ftypes, patch_size, patch_origin=(0, 0), savepath='./', 
                    prefix='window', num_patches=1):
    '''
    Takes an image file and creates a series of windows clipped from that image.
    The window slides across the original image at user defined intervals.
    Args:
        datapath: (str) the location of the original image(s).
        ftypes: (list) the filetypes to look for eg ['.tif'].
        patch_size: (tuple) the size of patch to cut.
        patch_origin: (tuple) the pixel location of the upper left of the first window default (0, 0).
        savepath: (str) the location to save to, default ./.
        num_patches: (int) the number of patches to create default 1.
    Returns:
    '''
    images = _find_image_files(datapath, ftypes)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    h=patch_origin[0]
    w=patch_origin[1]
    for count, im in enumerate(images):
        X = np.asarray(Image.open(im))
        for i in range(num_patches):
            y = X[h : h + patch_size[0], w + i : w + i + patch_size[1]]
            img = PIL.Image.fromarray(y)
            number = count * num_patches + i
            img.save( '%s/%s-%i.tiff' % (savepath, prefix, number))

