# -*- coding: utf-8 -*-
""":w

@author: Keith Butler
"""
import numpy as np
import threading
from .tools import normalise_discritise_data, build_list_images
from skimage.util import view_as_blocks
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from sklearn.preprocessing import normalize

def _crop_image(im, target_size):
    '''Given an image crop out the central region of a certain size. '''
    width = len(im)
    height = len(im[0])
    left = int((width - target_size[0])/2) 
    top = int((height - target_size[1])/2)
    right = int((width + target_size[0])/2)
    bottom = int((height + target_size[1])/2)
    cropped = [i[top:bottom] for i in im[left:right]]
    return cropped

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serialising call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generator_from_df(df, dummy_y, batch_size, target_size, debug=False,
                      debug_merged=False, resize_mode='scale', colormode='grayscale'):
    ''' 
    This generator creates one image per file and flows it to the model. It can be used to rescale/crop the input if needed. 

    Args:
        resize_mode: how to resize the image (scale/crop) (string)

    If features is not None, assume it is the path to a bcolz array
    that can be indexed by the same indexing of the input df.

    Assume input DataFrame df has columns 'imgpath' and 'target', where
    'imgpath' is full path to image file.

    https://github.com/fchollet/keras/issues/1627
    https://github.com/fchollet/keras/issues/1638

    Be forewarned if/when you modify this function: some errors will
    not be explicit, appearing only as a generic:

    ValueError: output of generator should be a tuple `(x, y, sample_weight)`
    or `(x, y)`. Found: None

    It usually means something in your infinite loop is not doing what
    you think it is, so the loop crashes and returns None.  Check your
    DataFrame in this function with various print statements to see if
    it is doing what you think it is doing.

    Again, error messages will not be too helpful here--if in doubt,
    print().
    '''

    nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)

    count = 1
    epoch = 0
    while 1:
       # df = df.sample(frac=1)  # frac=1 is same as shuffling df.     
        epoch += 1
        
        i, j = 0, batch_size
        # Mini-batches within epoch.
        mini_batches_completed = 0
        for _ in range(nbatches):
            sub = df.iloc[i:j]
            try:
                if resize_mode == 'scale':
                    X = np.array([(2 * (img_to_array(
                            load_img(f, color_mode=colormode, target_size=target_size)) / 255.0 - 0.5))
                            for f in sub.imgpath])
                    Y = dummy_y[i:j]
                    mini_batches_completed += 1
                    if debug:
                        print("X, y", sub.phases.iloc[1], Y[1])
                    yield X, Y
                else:
                   lst = []
                   for f in sub.imgpath:
                       im = 2 * (img_to_array(load_img(f, color_mode = colormode)) / 255.0 -0.5)
                       lst.append(np.asarray(_crop_image(im, target_size)))
                   X = np.array(lst)
                   Y = np.array(dummy_y[i:j])
                   if debug:
                       import matplotlib.pyplot as plt
                       fig, ax = plt.subplots(nrows=1, ncols=1)
                       ax.imshow(X[0, :, :, 0])
                       print(X[0, 0:50, 0:50, 0])
                       plt.savefig('debug/debug.png')
                       plt.close(fig)
                   mini_batches_completed += 1
                   yield X, Y
            except IOError:
                count -= 1
            i = j
            j += batch_size
            count += 1
            
@threadsafe_generator
def mask_patch_from_file(datapath, img_dir, mask_dir, patch_size, target_size, types=None,
                            debug=False, debug_merged=False, patch_range = [], 
                            batch_size = 1, normalise_images=False):
    """This generator creates sequential patches from an image file and flows each one to the model.
       The generator also creates masks of the same dimenstion and from the same location of the mask.
       Expects the directory structure
       Data
         |
         ----Images
         |
         ----Masks

       Args:
           datapath: path to the data
           img_dir: directory with the true images
           mask_dir: directory with the labelled masks
           patch_size: size of the patches to take
           target_size: the size to rad in the image at
           patch_range: list of the patches from the images to use for training (default = all)
           batch_size: (int) the number of images over which to average (default=1)
           normaise_image: (bool) should the values in the image be normalised? (default=False)

    """

    if target_size[0] % patch_size[0] != 0 or \
       target_size[1] % patch_size[1] != 0:
       print('Image and patch size are incommensurate, aborting.')
       return

    imgfiles = build_list_images(datapath + img_dir, types=types)
    maskfiles = build_list_images(datapath + mask_dir, types=types)
    if debug:
        print('Filename: {}'.format(imgfiles[0]))
        print('Maskname: {}'.format(maskfiles[0]))

    nbatches, n_skipped_per_epoch = divmod(len(imgfiles), batch_size)
    count = 1
    epoch = 0
    
    if len(patch_range) == 0:
        patch_range = np.arange(target_size[0]/patch_size[0]*target_size[1]/patch_size[1])
    n_height = int(target_size[0]/patch_size[0])
    n_width = int(target_size[1]/patch_size[1])
    while 1:
        epoch += 1
        # Mini-batches within epoch.
        mini_batches_completed = 0
        for patch in patch_range:
            x_loc = int(patch / n_height)  
            y_loc = int(patch % n_width)
            i, j = 0, batch_size
            for _ in range(nbatches):
                imgfiles_batch = imgfiles[i:j]
                maskfiles_batch = maskfiles[i:j]
                try:
                    # convert the patch index to image location
                    patch_list = []
                    mask_list = []
                    for i, sub in enumerate(imgfiles_batch):
                        mini_batches_completed += 1
                        X = np.asarray(Image.open(sub))
                        Y = np.asarray(Image.open(maskfiles_batch[i]))
                        patches = view_as_blocks(X, block_shape=patch_size)
                        mask_patches = view_as_blocks(Y, block_shape=patch_size)
                        patch = patches[x_loc][y_loc]
                        mask = mask_patches[x_loc][y_loc]
                        if normalise_images:
                            patch = normalize(patch, axis=1, norm='l1')
                        patch_list.append(np.resize(patch, (1, patch_size[0], 
                                                patch_size[1], 1)))
                        mask_list.append(np.resize(mask, (1, patch_size[0], 
                                                patch_size[1], 1)))
                    patch = sum([patch_list[i] for i in range(batch_size)]) / batch_size
                    mask = sum([mask_list[i] for i in range(batch_size)]) / batch_size
                    if debug:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(nrows=2, ncols=1)
                        patch_s = patch[0, :, :, 0]
                        mask_s = mask[0, :, :, 0]
                        ax[0].imshow(patch_s)
                        ax[1].imshow(mask_s)
                        plt.savefig('debug/debug_mask.png')
                        plt.close(fig)
                    yield normalise_discritise_data(patch, mask, debug=False)

                except IOError as err:
                    count -= 1
                i = j
                j += batch_size

@threadsafe_generator
def mask_patch_weights_from_file(datapath, img_dir, mask_dir, patch_size, target_size, weight_map,
                             types=None, debug=False, debug_merged=False, patch_range = []):
    """This generator creates sequential patches from an image file and flows each one to the model.
       The generator also creates masks of the same dimenstion and from the same location of the mask.
       This is the same as the patch_flow_from_file generator, but this is used with the weighted_umap.
       This works with the weighted loss function.
       Expects the directory structure
       Data
         |
         ----Images
         |
         ----Masks

       Args:
           datapath: path to the data
           img_dir: directory with the true images
           mask_dir: directory with the labelled masks
           patch_size: size of the patches to take
           target_size: the size to rad in the image at
           patch_range: list of the patches from the images to use for training (default = all)

    """

    imgfiles = build_list_images(datapath + img_dir, types=types)
    maskfiles = build_list_images(datapath + mask_dir, types=types)
    if debug:
        print('Filename: {}'.format(imgfiles[0]))
        print('Maskname: {}'.format(maskfiles[0]))

    batch_size = 1 # We are patching, so we go one image at a time
    nbatches, n_skipped_per_epoch = divmod(len(imgfiles), batch_size)
    count = 1
    epoch = 0
    
    if len(patch_range) == 0:
        patch_range = [0, target_size[0]/patch_size[0]*target_size[1]/patch_size[1]]

    while 1:
        epoch += 1
        
        i, j = 0, batch_size
        # Mini-batches within epoch.
        mini_batches_completed = 0
        for _ in range(nbatches):
            sub = imgfiles[i]
            sub_mask = maskfiles[i]
            try:
                X = np.asarray(Image.open(sub).convert('L'))
                print('Shape: ', X.shape)
                print('File: ', sub)
                Y = np.asarray(Image.open(sub_mask))
                patches = view_as_blocks(X, block_shape=patch_size)
                mask_patches = view_as_blocks(Y, block_shape=patch_size)
                mini_batches_completed += 1
                patches_completed = 0
                for k in range(patches.shape[0]):
                    for l in range(patches.shape[1]):
                        patch = patches[k][l]
                        mask = mask_patches[k][l]
                        patch = np.resize(patch, (1, patch_size[0], patch_size[1], 1))
                        mask = np.resize(mask, (1, patch_size[0], patch_size[1], 1))
                        if patches_completed in patch_range:
                            adjusted_patches = normalise_discritise_data(patch, mask, debug=False)
                            yield ([weight_map, adjusted_patches[0]], adjusted_patches[1])
                        patches_completed += 1
                
            except IOError as err:
                count -= 1
            i = j
            j += batch_size

@threadsafe_generator
def test_patch_flow_from_file(datapath, patch_size, target_size, 
                              debug=False, normalise_images=False):
    """This generator creates sequential patches from an image file and flows each one to the model.

       Args:
           datapath: path to the file to convert
           patch_size: size of the patches to take
           target_size: the size to rad in the image at

    """

    imgfiles = datapath

    X = np.asarray(Image.open(imgfiles))
    if normalise_images:
        X = X / 255
    patches = view_as_blocks(X, block_shape=patch_size)
    for k in range(patches.shape[0]):
        for l in range(patches.shape[1]):
            patch = patches[k][l]
            patch = np.resize(patch, (1, patch_size[0], patch_size[1], 1))
            yield patch
