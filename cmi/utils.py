"""Utility functions for reproducibility and TensorFlow configuration."""

import os
import numpy as np
import tensorflow as tf


def configure_tensorflow():
    """Configure TensorFlow to use CPU only."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def seed_everything(seed):
    """Fix random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

