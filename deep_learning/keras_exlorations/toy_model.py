
# This is for some explorations in keras API
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from os.path import join as opj


from scipy.io import loadmat, savemat

# we want reproducible results whenever possible
from numpy.random import seed
curPath = os.path.abspath(os.path.dirname(__file__))
data_path = opj(curPath, 'data')

mat_contents = loadmat(opj(data_path, 'training_EM_data.mat')) # samples must be scaled to [0,1]

print(mat_contents['trainingData'].shape) # (100, 224)

print(mat_contents.keys())

# print(mat_contents['m_idx'])

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# network parameters
original_dim, num_samples = mat_contents['trainingData'].shape
input_shape = (original_dim, num_samples, 1)
batch_size = int(mat_contents['batchSize'])
latent_dim = int(mat_contents['latent_dim'])
epochs = 50


x_train = mat_contents['trainingData'] # first dimension is training sample, second is input dim 
x_test  = x_train
y_train = np.zeros(num_samples)
y_test  = y_train
x_train = np.reshape(x_train, [1, original_dim, num_samples, 1])
x_test = np.reshape(x_train, [1, original_dim, num_samples, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
intermediate_dim1 = int(np.ceil(original_dim*1.2) + 5)  
intermediate_dim2 = int(max(np.ceil(original_dim/4), latent_dim+2) + 3)
intermediate_dim3 = int(max(np.ceil(original_dim/10), latent_dim+1))
activFun = mat_contents['actFunStr'][0]
inputs = Input(shape=input_shape, name='encoder_input')
x1 = Conv2D(filters=16,
                        kernel_size = 3,
                        activation = 'relu',
                        strides = 2,
                        padding = 'same')(inputs)
shape = K.int_shape(x1)
x1 = Flatten()(x1)
x2 = Dense(16, activation='relu')(x1)
z_mean = Dense(latent_dim, name='z_mean')(x2)
z_log_var = Dense(latent_dim, name='z_log_var')(x2)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

print(K.int_shape(z))
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


# decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x2 = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x2 = Reshape((shape[1], shape[2], shape[3]))(x2)
x1 = Conv2DTranspose(filters=16,
                     kernel_size=3,
                     activation='relu',
                     strides=2,
                     padding='same')(x2)
#outputs = Dense(original_dim, activation='sigmoid')(x1)
outputs = Conv2DTranspose(filters=1,
                         kernel_size=3,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x1)
# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
######## plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)


# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')







# m_sig = np.transpose(mat_contents['m_idx'])
# print(m_sig.shape)
# m_sig = np.reshape(m_sig, [1, original_dim, num_samples, 1])




