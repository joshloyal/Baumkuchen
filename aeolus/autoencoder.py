import keras.backend as K
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import regularizers
import numpy as np


def add_gaussian_noise(x, mean=0.0, variance=0.1, random_state=1234):
    rng = np.random.RandomState(random_state)
    return x + rng.normal(mean, variance, K.int_shape(x))


def AutoEncoder(input_dim, encoding_dim, add_noise=None, dropout_proba=None, activation='relu'):
    model_input = Input(shape=(input_dim,))

    if add_noise is not None:
        model_input = Lambda(add_noise)(model_input)

    encoded = Dense(encoding_dim, activation=activation,
                    activity_regularizer=regularizers.activity_l1(10e-5))(model_input)
    if dropout_proba:
        encoded = Dropout(dropout_proba)(encoded)

    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    if dropout_proba:
        decoded = Dropout(dropout_proba)(decoded)

    autoencoder = Model(input=model_input, output=encoded)
    autoencoder.compile('adadelta', loss='binary_crossentropy')

    return autoencoder


if __name__ == '__main__':
    model = AutoEncoder(input_dim=784, encoding_dim=32)
