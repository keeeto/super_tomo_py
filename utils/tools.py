import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from data_handeling.generators import test_patch_flow_from_file
import h5py
import imageio

def read_reconstruct_library(filepath):
    '''
    Reads in the data for testing the reconstruction CNN
    Args: 
        filepath: (str) the path to the files
    Returns:
        iamges: array of the images
        sinograms: array of sinograms
        nim: the number of images
    '''

    with h5py.File(filepath, 'r') as f:
        # List all groups
        a_group_key = list(f.keys())
        # Get the data
        images = []
        sinos = []
        nim = 0
        for key in a_group_key:
            if key == 'Images':
                images = np.array([list(f[key])])
                images = np.transpose(images, (3,1,2,0))
            elif key == 'Sinograms':
                sinograms = np.array([list(f[key])])
                sinograms = np.transpose(sinograms, (3,1,2,0))
            elif key == 'NImages':
                nim = np.array((f[key]))
        return(images, sinograms, nim)

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

def inference_binary_segmentation(datapath, patch_shape, img_shape, model,
                       file_prefix='binary_mask', savepath='.', 
		       fig_size=(20, 8), debug=False,
                       batch_size=1, normim=False):
    '''
    Function to plot and save an image of the binary mask.
    Args:
        datapath: the folder where the images to treat are located
	file_prefix: the prefix for the saved file name
	savepath: the path to the location to save the files
	patch_shape: pixel shape of the patch
	img_shape: pixel shape of the original image
	fig_size: the size of the images to produce
	model: the loaded Keras model
        batch_size: the batch size for running prediction model
        normim: normalise the images fed to the model?

    '''
    import matplotlib.pyplot as plt
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imgfiles = build_list_images(datapath, types=(['.png', '.tiff']))
    y_patches = int(img_shape[0] / patch_shape[0])
    x_patches = int(img_shape[1] / patch_shape[1])
    num_patches = x_patches * y_patches 
    print(imgfiles)

# Open the figure before the loop - so we overwrite and save time
    fig, ax = plt.subplots(nrows=1, ncols=1
                          , figsize=(20, 8))
# Load the images and run through the model
    for imnum, imfile in enumerate(imgfiles):
        testGene = test_patch_flow_from_file(imfile,
                    patch_shape, img_shape, debug=False,
                    normalise_images=normim)
        results = model.predict(testGene, 
                  num_patches, verbose=0)

# Convert from probabilities to binary mask
        if debug:
            print("Maximum mask predicted {0}.".format(np.max(results)))
            print("Minmum mask predicted {0}.".format(np.min(results)))
        for result in results:
            result[result > 0.5] = 1
            result[result <= 0.5] = 0

# Plot and save the results
# Rename some variables to make the following function readable
        h = patch_shape[0]
        w = patch_shape[1]
        rows = y_patches
        cols = x_patches
        imarray = np.array(results)
        imarray = imarray.reshape(rows, cols, h, w).swapaxes(1, 2) \
                  .reshape(h*rows, w*cols)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(np.squeeze(imageio.imread(imfile)), cmap='gray')	
        ax[1].imshow(imarray, cmap='gray')	
        for i in range(2):
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
        # TODO - save 8-bit byyinary image rahter than RGB
        plt.savefig(savepath + '{}_{:05d}.tif'.format(file_prefix, imnum))

    return

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
