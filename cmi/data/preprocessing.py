"""Data preprocessing utilities."""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler


def remove_gravity_from_acc(acc_data, rot_data):
    """Remove gravity component from accelerometer data.

    Args:
        acc_data: Accelerometer data (DataFrame or array)
        rot_data: Rotation quaternion data (DataFrame or array)

    Returns:
        Linear acceleration array without gravity
    """
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]

    return linear_accel


def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str],
                        scaler: StandardScaler):
    """Preprocess sequence data.

    Args:
        df_seq: Sequence DataFrame
        feature_cols: List of feature column names
        scaler: Fitted StandardScaler

    Returns:
        Preprocessed and scaled sequence array
    """
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    result = scaler.transform(mat).astype('float32')
    del mat
    return result

