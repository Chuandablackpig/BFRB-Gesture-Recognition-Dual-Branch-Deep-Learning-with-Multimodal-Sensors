"""Feature engineering functions."""

import numpy as np
import pandas as pd


def extract_imu_features(df):
    """Extract IMU features from raw data.

    Args:
        df: Input DataFrame with IMU data

    Returns:
        DataFrame with engineered IMU features
    """
    df = df.copy()

    # Base features
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    # Derivatives
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    # Linear acceleration features
    from cmi.data.preprocessing import remove_gravity_from_acc

    linear_accel_list = []
    for _, group in df.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        linear_accel_group = remove_gravity_from_acc(acc_data_group,
                                                     rot_data_group)
        linear_accel_list.append(
            pd.DataFrame(linear_accel_group,
                        columns=['linear_acc_x', 'linear_acc_y',
                                'linear_acc_z'],
                        index=group.index))

    df_linear_accel = pd.concat(linear_accel_list)
    df = pd.concat([df, df_linear_accel], axis=1)

    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 +
                                   df['linear_acc_y']**2 +
                                   df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = (df.groupby('sequence_id')['linear_acc_mag']
                                 .diff().fillna(0))

    return df


def extract_tof_features(df):
    """Extract aggregated TOF features.

    Args:
        df: Input DataFrame with TOF pixel data

    Returns:
        DataFrame with aggregated TOF features
    """
    df = df.copy()

    for i in range(1, 6):
        pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
        tof_sensor_data = df[pixel_cols_tof].replace(-1, np.nan)
        df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
        df[f'tof_{i}_std'] = tof_sensor_data.std(axis=1)
        df[f'tof_{i}_min'] = tof_sensor_data.min(axis=1)
        df[f'tof_{i}_max'] = tof_sensor_data.max(axis=1)

    return df


def build_feature_columns(df):
    """Build final feature column list.

    Args:
        df: DataFrame with all features

    Returns:
        Tuple of (feature_columns, imu_dim, tof_dim)
    """
    imu_cols_base = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
    imu_cols_base.extend([c for c in df.columns
                          if c.startswith('rot_') and
                          c not in ['rot_angle', 'rot_angle_vel']])

    imu_engineered_features = [
        'acc_mag', 'rot_angle',
        'acc_mag_jerk', 'rot_angle_vel',
        'linear_acc_mag', 'linear_acc_mag_jerk'
    ]
    imu_cols = imu_cols_base + imu_engineered_features
    imu_cols = list(dict.fromkeys(imu_cols))

    thm_cols_original = [c for c in df.columns if c.startswith('thm_')]
    tof_aggregated_cols_template = []

    for i in range(1, 6):
        tof_aggregated_cols_template.extend([
            f'tof_{i}_mean', f'tof_{i}_std',
            f'tof_{i}_min', f'tof_{i}_max'])

    final_feature_cols = (imu_cols + thm_cols_original +
                         tof_aggregated_cols_template)

    imu_dim_final = len(imu_cols)
    tof_thm_aggregated_dim_final = (len(thm_cols_original) +
                                    len(tof_aggregated_cols_template))

    return final_feature_cols, imu_dim_final, tof_thm_aggregated_dim_final

