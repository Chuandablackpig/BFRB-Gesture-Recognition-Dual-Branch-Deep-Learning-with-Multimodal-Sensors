"""Training pipeline implementation."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from cmi.config import (
    RAW_DIR, EXPORT_DIR, BATCH_SIZE, PAD_PERCENTILE, WD, MIXUP_ALPHA,
    EPOCHS, PATIENCE
)
from cmi.models import build_two_branch_model, MixupGenerator
from cmi.data import extract_imu_features, build_feature_columns
from cmi_2025_metric_copy_for_import import CompetitionMetric


def train_model():
    """Train the two-branch model."""
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")
    train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
    df_for_groups = pd.merge(df.copy(), train_dem_df, on='subject', how='left')

    # Encode labels
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
    gesture_classes = le.classes_

    # Feature engineering
    print("  Calculating base engineered IMU features...")
    df = extract_imu_features(df)

    # Build feature columns (before TOF aggregation)
    final_feature_cols, imu_dim_final, tof_thm_aggregated_dim_final = (
        build_feature_columns(df))

    print(f"  IMU {imu_dim_final} | THM+TOF {tof_thm_aggregated_dim_final} | "
          f"total {len(final_feature_cols)} features")
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))

    # Build sequences with TOF aggregation per sequence
    print("  Building sequences with aggregated TOF...")
    seq_gp = df.groupby('sequence_id')

    all_steps_for_scaler_list = []
    X_list_unscaled, y_list_int_for_stratify, lens = [], [], []

    for seq_id, seq_df_orig in seq_gp:
        seq_df = seq_df_orig.copy()

        # Extract TOF features for this sequence
        for i in range(1, 6):
            pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
            tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)
            seq_df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
            seq_df[f'tof_{i}_std'] = tof_sensor_data.std(axis=1)
            seq_df[f'tof_{i}_min'] = tof_sensor_data.min(axis=1)
            seq_df[f'tof_{i}_max'] = tof_sensor_data.max(axis=1)

        mat_unscaled = (seq_df[final_feature_cols].ffill().bfill()
                       .fillna(0).values.astype('float32'))

        all_steps_for_scaler_list.append(mat_unscaled)
        X_list_unscaled.append(mat_unscaled)
        y_list_int_for_stratify.append(seq_df['gesture_int'].iloc[0])
        lens.append(len(mat_unscaled))

    # Fit scaler
    print("  Fitting StandardScaler...")
    all_steps_concatenated = np.concatenate(all_steps_for_scaler_list, axis=0)
    scaler = StandardScaler().fit(all_steps_concatenated)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    del all_steps_for_scaler_list, all_steps_concatenated

    # Scale and pad sequences
    print("  Scaling and padding sequences...")
    X_scaled_list = [scaler.transform(x_seq) for x_seq in X_list_unscaled]
    del X_list_unscaled

    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)

    X = pad_sequences(X_scaled_list, maxlen=pad_len, padding='post',
                     truncating='post', dtype='float32')
    del X_scaled_list

    y_int_for_stratify = np.array(y_list_int_for_stratify)
    y = to_categorical(y_int_for_stratify, num_classes=len(le.classes_))

    # Split data
    print("  Splitting data and preparing for training...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=82, stratify=y_int_for_stratify)

    # Compute class weights
    cw_vals = compute_class_weight('balanced',
                                   classes=np.arange(len(le.classes_)),
                                   y=y_int_for_stratify)
    class_weight = dict(enumerate(cw_vals))

    # Build and compile model
    model = build_two_branch_model(pad_len, imu_dim_final,
                                   tof_thm_aggregated_dim_final,
                                   len(le.classes_), wd=WD)

    steps = len(X_tr) // BATCH_SIZE
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        5e-4, first_decay_steps=15 * steps)

    model.compile(optimizer=Adam(lr_sched),
                  loss=tf.keras.losses.CategoricalCrossentropy(
                      label_smoothing=0.1),
                  metrics=['accuracy'])

    # Training
    train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE,
                               alpha=MIXUP_ALPHA)
    cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                      verbose=1, monitor='val_accuracy', mode='max')

    print("  Starting model training...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=(X_val, y_val),
             class_weight=class_weight, callbacks=[cb], verbose=1)

    model.save(EXPORT_DIR / "gesture_two_branch_mixup.h5")
    print("✔ Training done – artefacts saved in", EXPORT_DIR)

    # Evaluation
    preds_val = model.predict(X_val).argmax(1)
    true_val_int = y_val.argmax(1)

    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[true_val_int]}),
        pd.DataFrame({'gesture': le.classes_[preds_val]}))
    print("Hold‑out H‑F1 =", round(h_f1, 4))

