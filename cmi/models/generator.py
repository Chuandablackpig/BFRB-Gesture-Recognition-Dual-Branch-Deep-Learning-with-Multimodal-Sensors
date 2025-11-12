"""Data generators for training."""

import numpy as np
from tensorflow.keras.utils import Sequence


class MixupGenerator(Sequence):
    """Mixup data augmentation generator."""

    def __init__(self, X, y, batch_size, alpha=0.2):
        """Initialize MixupGenerator.

        Args:
            X: Input features array
            y: Target labels array
            batch_size: Batch size for training
            alpha: Mixup alpha parameter
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(self.X))
        self._check_data_types()

    def _check_data_types(self):
        """Validate data types."""
        if not isinstance(self.X, np.ndarray) or not isinstance(self.y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        if self.X.dtype != 'float32':
            self.X = self.X.astype('float32')
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples")

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

