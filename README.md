# CMI 2025 BFRB Detection

Body Focused Repetitive Behaviors (BFRB) Detection project for Kaggle CMI 2025 Challenge.

## Project Structure

```
CMI/
├── cmi/                    # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration constants
│   ├── utils.py            # Utility functions
│   ├── models/             # Model architecture
│   │   ├── __init__.py
│   │   ├── layers.py       # Custom layers and blocks
│   │   ├── model.py        # Model builder
│   │   └── generator.py    # Data generators
│   ├── data/               # Data processing
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Data preprocessing
│   │   └── features.py     # Feature engineering
│   ├── training/           # Training pipeline
│   │   ├── __init__.py
│   │   └── train.py
│   └── inference/          # Inference
│       ├── __init__.py
│       └── predict.py
├── main.py                 # Main entry point
├── cmi_2025_metric_copy_for_import.py  # Evaluation metric
└── README.md
```

## Usage

### Training

Set `TRAIN = True` in `cmi/config.py` and run:

```bash
python main.py
```

### Inference

Set `TRAIN = False` in `cmi/config.py` and run:

```bash
python main.py
```

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Polars
- scikit-learn
- scipy

## Code Style

This project follows PEP 8 code style guidelines.

