import  numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import Model
from tqdm import tqdm
from skimage.transform import resize

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def dimension_reducer(images, in_dim=(224, 224), n_components_pca=50, n_components_tsne=2,
                     learning_rate=200, early_exaggeration=4, perplexity=50, n_iter=5000):
    '''
    Function to take a dataset of images and reduce dimensions and return their distrubtion in a reduced
    dimensional space.
    Args:
        images: An array of images (shape=(number, width, height, channels))
        in_dim: The dimensionality of images to feed to the reducer (this ensures that all images are of the same size before dim reduction)
        n_components_pca: integer, the number of components to take from the PCA to feed to tSNE
        n_components_tsne: integer, the final dimensionality required
        learning_rate: integer, the tSNE learning rate
        early_exaggeration: integer, the early exaggeration for tSNE
        perplexity: integer,  the perplexity for tSNE
        n_iter: integer, the number of tSNE iterations  
    Returns:
        An n-dimensional array, where n is the number of dimensions requested
        
    '''

    nim = images.shape[0]
    vol_resized = np.zeros((in_dim[0],in_dim[1], nim))
    
    for ii in range(nim):      
        img = images[ii,:,:,0]
        vol_resized[:,:,ii] = resize(img, in_dim, anti_aliasing=True)    
    vol_resized = np.array([vol_resized]).transpose(1,2,3,0)
    vol_resized = tf.convert_to_tensor(vol_resized)
    vol_resized = tf.image.grayscale_to_rgb(vol_resized)
    vol_resized = np.array([np.array(vol_resized)])
    vol_resized = tf.convert_to_tensor(vol_resized)

    model = resnet50.ResNet50(weights='imagenet')
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(inputs=model.input,
                                outputs=model.get_layer(layer_name).output)

    img = np.array(images[0,:,:,0])
    x = resize(img, in_dim, anti_aliasing=True)
    x = np.array([x]).transpose(1,2,0)
    x = tf.convert_to_tensor(x)
    x = tf.image.grayscale_to_rgb(x)
    x = np.array([np.array(x)])
    x = resnet50.preprocess_input(x)
    intermediate_output = intermediate_layer_model.predict(x)
    vectors = np.zeros((nim, intermediate_output.shape[1]))
    for i in tqdm(range(nim)):
        
        x = vol_resized[:,:,:,i,:]
        x = resnet50.preprocess_input(x)
        intermediate_output = intermediate_layer_model.predict(x)
        vectors[i] = intermediate_output[0]
    X = vectors[:nim]
    
    pca_50 = PCA(n_components=n_components_pca)
    pca_result_50 = pca_50.fit_transform(X)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
    print(np.shape(pca_result_50))
    
    # for complex data which has shape to it, run t-sne with 4 components to see 3D clustering
    tsne = TSNE(n_components=n_components_tsne, verbose=1, learning_rate=learning_rate,
                early_exaggeration=early_exaggeration, perplexity=perplexity,
                n_iter=n_iter, method='exact') 
    tsne_result = tsne.fit_transform(pca_result_50)
    
    return StandardScaler().fit_transform(tsne_result)
