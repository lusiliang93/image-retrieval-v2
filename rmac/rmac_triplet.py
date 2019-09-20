from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import h5py

from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map

import scipy.io
import utils

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import os
from os import listdir
from os.path import join
import keras

def triplet_loss(y_true, y_pred):
        margin = K.constant(1)
        ''' soft margin
        dist = y_pred[:,0,0] - y_pred[:,1,0]
        margin = np.log(1+np.exp(dist))
        '''
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):

    # Load ResNet50
    # vgg16_model = ResNet50(weights='imagenet')
    vgg16_model = keras.models.load_model('../triplet_loss_resnet50.h5',custom_objects={'triplet_loss':triplet_loss})
    vgg16_model.summary()

    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # Must check dimension when using finetuned model
    resnet_model = vgg16_model.layers[-4]
    # Second to last conv2 layer
    x = RoiPooling([1], num_rois)([resnet_model.layers[-12].output, in_roi])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)


    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([resnet_model.get_input_at(0), in_roi], rmac_norm)


    # Load PCA weights
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])

    # save model into .h5
    return model

def rmac_holidays(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[1], x.shape[2])
    # warning can be ignored
    regions = rmac_regions(Wmap, Hmap, 3)
    print('Loading RMAC model...')
    model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
    # Compute RMAC vector
    print('Extracting RMAC from image...')
    RMAC1 = model.predict([x, np.expand_dims(regions, axis=0)])
    return RMAC1


if __name__ == "__main__":
    '''
    data_dir_query = '../query'
    datalist_query = [join(data_dir_query, f) for f in listdir(data_dir_query)]
    # Mac will generate another file...
    datalist_query = datalist_query[1:]
    features = []
    for i, img1_path in enumerate(datalist_query):
        img1 = image.load_img(img1_path, target_size=(224, 224))
        RMAC1 = rmac_holidays(img1)
        RMAC1 = RMAC1.flatten()
        features.append(RMAC1)

    h5f = h5py.File('embeddeing.h5','w')
    h5f.create_dataset('embedding',data=features)
    h5f.close()
    '''
    img1 = image.load_img('100001.jpg', target_size=(224, 224))
    RMAC1 = rmac_holidays(img1)
    print(RMAC1)