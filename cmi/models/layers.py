"""Custom layers and blocks for the neural network."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    GlobalAveragePooling1D, Dense, Multiply, Reshape, Lambda
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def time_sum(x):
    """Sum over time axis."""
    return K.sum(x, axis=1)


def squeeze_last_axis(x):
    """Squeeze last axis."""
    return tf.squeeze(x, axis=-1)


def expand_last_axis(x):
    """Expand last axis."""
    return tf.expand_dims(x, axis=-1)


def se_block(x, reduction=8):
    """Squeeze-and-Excitation attention block."""
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])


def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    """Residual SE-CNN block."""
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                         kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x


def attention_layer(inputs):
    """Temporal attention layer."""
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

