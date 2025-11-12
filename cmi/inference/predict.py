"""Prediction and inference functions."""

import numpy as np
import pandas as pd
import joblib
import polars as pl
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

from cmi.config import PRETRAINED_DIR
from cmi.models.layers import (
    time_sum, squeeze_last_axis, expand_last_axis,
    se_block, residual_se_cnn_block, attention_layer
)
from cmi.data.preprocessing import remove_gravity_from_acc
from cmi.data.features import extract_imu_features, extract_tof_features


# Global variables for loaded model artifacts
_model = None
_scaler = None
_pad_len = None
_gesture_classes = None
_final_feature_cols = None


def load_model_artifacts():
    """Load pretrained model and artifacts."""
    global _model, _scaler, _pad_len, _gesture_classes, _final_feature_cols

    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    _final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy",
                                  allow_pickle=True).tolist()
    _pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    _scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    _gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy",
                               allow_pickle=True)

    temp_imu_cols = [c for c in _final_feature_cols
                    if c.startswith('acc_') or c.startswith('rot_')]
    imu_dim_final = len(temp_imu_cols)
    tof_thm_aggregated_dim_final = len(_final_feature_cols) - imu_dim_final

    custom_objs = {
        'time_sum': time_sum,
        'squeeze_last_axis': squeeze_last_axis,
        'expand_last_axis': expand_last_axis,
        'se_block': se_block,
        'residual_se_cnn_block': residual_se_cnn_block,
        'attention_layer': attention_layer,
    }
    _model = load_model(PRETRAINED_DIR / "gesture_two_branch_mixup.h5",
                       compile=False, custom_objects=custom_objs)
    print("  Model, scaler, feature_cols, pad_len loaded")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Predict gesture for a sequence.

    Args:
        sequence: Sequence data as Polars DataFrame
        demographics: Demographics data (unused but required by interface)

    Returns:
        Predicted gesture class name
    """
    global _model, _scaler, _pad_len, _gesture_classes, _final_feature_cols

    if _model is None:
        load_model_artifacts()

    df_seq = sequence.to_pandas()

    # Feature engineering
    df_seq = extract_imu_features(df_seq)
    df_seq = extract_tof_features(df_seq)

    # Handle missing columns
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    rot_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w']

    if not all(col in df_seq.columns for col in acc_cols + rot_cols):
        print("Warning: Missing raw acc/rot columns. Using raw acc as linear.")
        df_seq['linear_acc_x'] = df_seq.get('acc_x', 0)
        df_seq['linear_acc_y'] = df_seq.get('acc_y', 0)
        df_seq['linear_acc_z'] = df_seq.get('acc_z', 0)
    else:
        acc_data_seq = df_seq[acc_cols]
        rot_data_seq = df_seq[rot_cols]
        linear_accel_seq_arr = remove_gravity_from_acc(acc_data_seq,
                                                       rot_data_seq)
        df_seq['linear_acc_x'] = linear_accel_seq_arr[:, 0]
        df_seq['linear_acc_y'] = linear_accel_seq_arr[:, 1]
        df_seq['linear_acc_z'] = linear_accel_seq_arr[:, 2]

    # Handle optional features
    if 'tof_range_across_sensors' in _final_feature_cols:
        tof_mean_cols = [f'tof_{i}_mean' for i in range(1, 6)
                        if f'tof_{i}_mean' in df_seq.columns]
        thm_cols = [f'thm_{i}' for i in range(1, 6)
                   if f'thm_{i}' in df_seq.columns]

        if tof_mean_cols:
            tof_values = df_seq[tof_mean_cols]
            df_seq['tof_range_across_sensors'] = (tof_values.max(axis=1) -
                                                 tof_values.min(axis=1))
            df_seq['tof_std_across_sensors'] = tof_values.std(axis=1)
        else:
            df_seq['tof_range_across_sensors'] = 0
            df_seq['tof_std_across_sensors'] = 0

        if thm_cols:
            thm_values = df_seq[thm_cols]
            df_seq['thm_range_across_sensors'] = (thm_values.max(axis=1) -
                                                 thm_values.min(axis=1))
            df_seq['thm_std_across_sensors'] = thm_values.std(axis=1)
        else:
            df_seq['thm_range_across_sensors'] = 0
            df_seq['thm_std_across_sensors'] = 0

    # Build feature matrix
    df_seq_final_features = pd.DataFrame(index=df_seq.index)
    for col_name in _final_feature_cols:
        if col_name in df_seq.columns:
            df_seq_final_features[col_name] = df_seq[col_name]
        else:
            print(f"Warning: Feature '{col_name}' not found. Filling with 0.")
            df_seq_final_features[col_name] = 0

    # Preprocess and predict
    mat_unscaled = df_seq_final_features.ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = _scaler.transform(mat_unscaled)
    pad_input = pad_sequences([mat_scaled], maxlen=_pad_len, padding='post',
                              truncating='post', dtype='float32')

    idx = int(_model.predict(pad_input, verbose=0).argmax(1)[0])
    return str(_gesture_classes[idx])

