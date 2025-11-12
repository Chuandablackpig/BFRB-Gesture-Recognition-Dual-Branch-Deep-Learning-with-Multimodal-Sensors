"""Model architecture definition."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense, Concatenate, GRU, GaussianNoise, Lambda
)
from tensorflow.keras.regularizers import l2

from cmi.models.layers import (
    residual_se_cnn_block,
    attention_layer
)


def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    """Build two-branch neural network model.

    Args:
        pad_len: Padded sequence length
        imu_dim: IMU feature dimension
        tof_dim: TOF/thermal feature dimension
        n_classes: Number of output classes
        wd: Weight decay parameter

    Returns:
        Compiled Keras model
    """
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # TOF/thermal light branch
    x2 = Conv1D(64, 3, padding='same', use_bias=False,
                kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False,
                kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.2)(x2)

    # Merge branches
    merged = Concatenate()([x1, x2])
    xa = Bidirectional(LSTM(128, return_sequences=True,
                            kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True,
                           kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)

    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    # Final dense layers
    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(drop)(x)
    out = Dense(n_classes, activation='softmax',
                kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

