#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main entry point for CMI 2025 BFRB Detection."""

import os
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

from cmi.utils import configure_tensorflow, seed_everything
from cmi.config import TRAIN, SEED
from cmi.training import train_model
from cmi.inference import load_model_artifacts, predict
import kaggle_evaluation.cmi_inference_server

# Configure TensorFlow
configure_tensorflow()
print("▶ imports ready · tensorflow", tf.__version__)

# Set random seed
seed_everything(SEED)
print(f"Random seed fixed to {SEED}")

if TRAIN:
    train_model()
else:
    load_model_artifacts()
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                'data/test.csv',
                'data/test_demographics.csv',
            )
        )

