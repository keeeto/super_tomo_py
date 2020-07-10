# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:45:53 2020

@author: Antony
"""

import matplotlib.pyplot as plt
import time
from skimage.draw import random_shapes
import numpy as np
import astra

def cirmask(im, npx=0):
    
    """
    
    Apply a circular mask to the image
    
    """
    
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2));
    
    dim =  im.shape
    if len(dim)==2:
        im = np.where(r>np.floor(sz/2) - npx,0,im)
    elif len(dim)==3:
        for ii in range(0,dim[2]):
            im[:,:,ii] = np.where(r>np.floor(sz/2),0,im[:,:,ii])
    return(im)

#%% Create a test image

start=time.time()
sz=64; min_shapes=3; max_shapes=10; min_size=2; max_size=10

image, _ = random_shapes((sz, sz), min_shapes=min_shapes, max_shapes=max_shapes, multichannel=False,
                         min_size=min_size, max_size=max_size, allow_overlap=True)
image = np.where(image==255, 1, image)
image = cirmask(image,5)
image = image/np.max(image)


ct = 2**8
image = np.random.poisson(lam=(image)*ct, size=None)/ct
image = image/np.max(image)

print((time.time()-start))

plt.figure(1);plt.clf();plt.imshow(image, cmap='jet');plt.show();

#%% Perform first a test for sinogram creation and image reconstruction

npr = image.shape[0] # Number of projections

# Create a basic square volume geometry
vol_geom = astra.create_vol_geom(image.shape[0], image.shape[0])
# Create a parallel beam geometry with 180 angles between 0 and pi, and image.shape[0] detector pixels of width 1.
proj_geom = astra.create_proj_geom('parallel', 1.0, int(1.0*image.shape[0]), np.linspace(0,np.pi,npr,False))
# Create a sinogram using the GPU.
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

start=time.time()
sinogram_id, sinogram = astra.create_sino(image, proj_id)
print((time.time()-start))

plt.figure(1);plt.clf();plt.imshow(sinogram, cmap='jet');plt.show();

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
# cfg = astra.astra_dict('SIRT_CUDA')
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['option'] = { 'FilterType': 'shepp-logan' }

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# Get the result
start=time.time()
rec = astra.data2d.get(rec_id)
print((time.time()-start))

rec = np.where(rec<0, 0, rec)
rec = cirmask(rec)

plt.figure(2);plt.clf();plt.imshow(np.concatenate((rec,image),axis=1), cmap='jet');
plt.colorbar();
plt.show();

astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)

#%% Perform a test using the SampleGen module

import os
from pathlib import Path
p = Path("C:\\Users\\Antony\\Documents\\GitHub\\NNs_in_Tensorflow2\\Libraries")
p = Path("C:\\Users\\Simon\\Documents\\GitHub\\NNs_in_Tensorflow2\\Libraries")
os.chdir(p)

from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
import SampleGen as sg

sml = sg.random_sample()

sz=64; min_shapes=3; max_shapes=10; min_size=2; max_size=10

sml.set_pars( sz=sz, min_shapes=min_shapes, max_shapes=max_shapes, min_size = min_size, max_size=max_size)
sml.create_image()

ct = 2**8
sml.im = np.random.poisson(lam=(sml.im)*ct, size=None)/ct
sml.im = sml.im/np.max(sml.im)

# Apply a filter
sml.im = gaussian_filter(sml.im, sigma=1)
# sml.im = uniform_filter(sml.im, size=3)
# sml.im = uniform_filter(sml.im, size=5)

sml.im = cirmask(sml.im,5)

plt.figure(1);plt.clf();plt.imshow(sml.im, cmap='jet');plt.colorbar();plt.show();

sml.create_sino_geo()
sml.create_sino()

plt.figure(2);plt.clf();plt.imshow(sml.sinogram, cmap='jet');plt.colorbar();plt.show();


#%% Create a library


nims = 100000
filtering = 1
noise = 1
scaling = 0

sz=32; min_shapes=3; max_shapes=10; min_size=3; max_size=10

ct = 2**8

start=time.time()

im = np.zeros((sz,sz,nims))
s = np.zeros((sz,sz,nims))

sf = np.random.rand(nims,)*10.0**-np.random.uniform(0, 2, nims)

for ii in range(nims):
 
    sml.set_pars( sz=sz, min_shapes=min_shapes, max_shapes=max_shapes, min_size = min_size, max_size=max_size)
    
    image = sml.create_image()
    
    image = cirmask(image,5)

    
    if np.max(image)>0:
        image = image/np.max(image)
    
    if noise == True:
        image = np.random.poisson(lam=(image)*ct, size=None)/ct
        if np.max(image)>0:
            image = image/np.max(image)

    if filtering == True:
        image = gaussian_filter(image, sigma=1)
        if np.max(image)>0:
            image = image/np.max(image)        
    
    if scaling == True:
        image = image * sf[ii]
    
    sml.im = image
    
    im[:,:,ii] = image
    
    s[:,:,ii] = sml.create_sino()
    
    if np.mod(ii, 100) == 0:    
        print(ii)

print((time.time()-start))

    # sml.astraclean()

#%%

for ii in range(0, nims):

    
    plt.figure(2);plt.clf();
    
    plt.imshow(im[:,:,ii], cmap = 'jet')
    
    plt.colorbar();
    
    plt.show()

    plt.pause(1)

#%%
import h5py

p = Path('./')

fn = Path("%s\\shapes_random_noise_%dpx_norm.h5" %(p, sz))

h5f = h5py.File(fn, "w")

h5f.create_dataset('Sinograms', data = s)
h5f.create_dataset('Images', data = im)
h5f.create_dataset('ScaleFactor', data = sf)
h5f.create_dataset('Noise', data = noise)
h5f.create_dataset('NImages', data = nims)
h5f.create_dataset('ImageSize', data = sz)

h5f.close()

