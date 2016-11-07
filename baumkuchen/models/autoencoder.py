import keras.backend as K
from keras.layers import Input, Dense, Lambda, Dropout
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras import regularizers
import numpy as np


def noise_output_shape(input_shape):
    return tuple(input_shape)


def gaussian_noise(x, mean=0.0, std=0.1, random_state=1234):
    return x + K.random_normal(K.shape(x), mean=mean, std=std, seed=random_state)


def AutoEncoder(input_dim, encoding_dim, add_noise=None, dropout_proba=None, l1=1e-4):
    model_input = Input(shape=(input_dim,))

    if add_noise is not None:
        x = Lambda(add_noise, output_shape=noise_output_shape)(model_input)
    else:
        x = model_input

    if l1 is not None:
        encoded = Dense(encoding_dim, activation='relu',
                        activity_regularizer=regularizers.activity_l1(l1))(x)
    else:
        encoded = Dense(encoding_dim, activation='relu')(x)

    if dropout_proba:
        encoded = Dropout(dropout_proba)(encoded)

    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input=model_input, output=decoded)
    autoencoder.compile(optimizer='adadelta',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    return autoencoder


if __name__ == '__main__':
    model = AutoEncoder(input_dim=784, encoding_dim=32)
