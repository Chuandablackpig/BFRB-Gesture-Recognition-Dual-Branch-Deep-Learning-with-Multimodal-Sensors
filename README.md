# BFRB Gesture Recognition: Dual-Branch Deep Learning with Multimodal Sensors
A Kaggle competition reproduction project for **Body-Focused Repetitive Behaviors (BFRB)** detection using wrist-worn multimodal sensor data. The project implements a dual-branch deep learning model to distinguish BFRB gestures (e.g., hair pulling, scratching) from non-BFRB daily actions with high accuracy.

## 🌟 Core Features
- **Multimodal Sensor Fusion**: Integrates IMU (motion), thermopile (thermal), and TOF (spatial) data to solve behavioral ambiguity of single-sensor systems.
- **Dual-Branch Architecture**: Residual SE-CNN for IMU motion features + lightweight CNN for TOF/thermopile data, enhanced by Bi-LSTM/Bi-GRU temporal modeling.
- **Robust Data Preprocessing**: Includes quaternion-based gravity removal, TOF missing value imputation, and engineered features (e.g., acceleration magnitude, angular velocity).
- **Generalization Enhancement**: Uses Mixup data augmentation to adapt to diverse user populations (ages 8–45, different handedness).
- **CPU-Compatible**: Full TensorFlow CPU configuration for easy deployment without GPU dependencies.

## 📊 Dataset
The dataset is collected via the **Helios wrist-worn device** (50 Hz sampling rate) and includes:
- **Sensor Data**:
  - IMU: 3-axis acceleration + 4-dimensional quaternion orientation.
  - Thermopile: 5 non-contact infrared sensors (surface temperature measurement).
  - TOF: 5 sensors with 8×8 pixel grids (spatial proximity mapping).
- **Annotations**: 18 gesture classes (8 BFRB + 10 non-BFRB), 4 postures (sitting, supine, etc.), and participant demographics (32 subjects: 18 adults, 14 children).
- **Data Structure**: Sequences divided into Transition → Pause → Gesture phases for fine-grained modeling.

## 🛠️ Installation
### Dependencies
Install required packages via pip:
```bash
pip install tensorflow numpy pandas polars scikit-learn scipy joblib
```

## Data Format & Example

The model expects **time-series data files** (`train.csv`, `test.csv`) where each row represents one timestamp, and all rows with the same `sequence_id` form a single gesture sequence.

### 🧭 Expected Columns

#### **IMU (Inertial Measurement Unit)**
| Column | Description |
|--------|--------------|
| `acc_x`, `acc_y`, `acc_z` | Raw accelerometer readings |
| `rot_w`, `rot_x`, `rot_y`, `rot_z` | Quaternion rotation components |

#### **TOF (Time-of-Flight) Sensors**
| Column Pattern | Description |
|----------------|-------------|
| `tof_1_v0` … `tof_1_v63` | 64 pixel readings for TOF sensor 1 |
| `tof_2_v0` … `tof_2_v63` | 64 pixel readings for TOF sensor 2 |
| ... | Up to `tof_5_v63` (5 sensors × 64 = 320 total pixels) |

#### **Thermopile Sensors**
| Column | Description |
|--------|--------------|
| `thm_1` … `thm_5` | 5 thermal sensor readings |

#### **Labels & Metadata**
| Column | Description |
|--------|--------------|
| `gesture` | Gesture label string (e.g., `"Hair Pulling"`) |
| `subject` | Participant ID |
| `sequence_id` | Unique identifier for one gesture sequence |

> ⚠️ Missing columns will be filled with zeros and warnings will be printed.  
> For best accuracy, ensure all fields are present and formatted consistently.

---

## Preprocessing & Feature Engineering

The preprocessing pipeline includes multiple steps that clean, enrich, and normalize the multimodal sensor data before feeding it into the model.

### 🧮 IMU Feature Engineering
- **Magnitude:**  
  \[
  acc\_mag = \sqrt{acc_x^2 + acc_y^2 + acc_z^2}
  \]
- **Rotation Angle:**  
  \[
  rot\_angle = 2 \times \arccos(rot_w.clip(-1, 1))
  \]
- **Derivatives:**  
  - `acc_mag_jerk` (rate of change of acceleration magnitude)  
  - `rot_angle_vel` (rotation velocity)

### 🌍 Gravity Removal
- Uses quaternion-based rotation (`remove_gravity_from_acc()`) to remove the gravity vector from raw acceleration, producing **linear acceleration** components (`linear_acc_x`, `linear_acc_y`, `linear_acc_z`).

### 📊 TOF Feature Aggregation
- Each TOF sensor’s 64-pixel map is summarized into **four statistical descriptors**:
  - `mean`
  - `std`
  - `min`
  - `max`
- 5 sensors × 4 statistics = **20 total TOF features**

### ⚙️ Missing Value Handling
- Forward-fill (`ffill`), backward-fill (`bfill`), and zero-fill for missing data  
- TOF pixel values of `-1` are treated as `NaN` before aggregation  

### 📏 Standardization
- A `StandardScaler` is fitted on all time steps of the training data.  
- Saved to: `output/scaler.pkl`  
- Used during both training and inference for normalization.

### 🧩 Sequence Padding
- All gesture sequences are padded (or truncated) to the **95th percentile of sequence length** (`pad_len`) using Keras `pad_sequences`, ensuring consistent input shape across batches.

---

## Model Architecture

Defined in **`build_two_branch_model()`**, the model is a dual-branch architecture that processes IMU and TOF/Thermopile features separately before fusion.

### 🧠 Input
- Shape: `(pad_len, imu_dim + tof_dim)`  
- Split internally into:
  - **IMU branch:** first `imu_dim` features  
  - **TOF branch:** remaining `tof_dim` features

---

### 🔹 IMU Branch
Two **Residual SE-CNN Blocks**:
- Each block includes:
  - Conv1D ×2 → BatchNorm → ReLU → SE block → Residual Add → MaxPool → Dropout
  - **Block 1:** filters = 64, kernel = 3  
  - **Block 2:** filters = 128, kernel = 5

---

### 🔸 TOF Branch
Two **Convolutional Blocks**:
  -Conv1D(64) → BN → ReLU → MaxPooling → Dropout
  -Conv1D(128) → BN → ReLU → MaxPooling → Dropout

---

### 🔗 Fusion & Temporal Modeling
After concatenating both branches:
- **Parallel RNN streams:**
  - `Bidirectional(LSTM(128, return_sequences=True))`
  - `Bidirectional(GRU(128, return_sequences=True))`
  - `GaussianNoise(0.09)` + `Dense(16, activation='elu')`
- Outputs from all three are concatenated → `Dropout(0.4)` → **Attention Layer**

---

### 🧩 Classification Head

-Dense(256) → BN → ReLU → Dropout(0.5)
-Dense(128) → BN → ReLU → Dropout(0.3)
-Dense(n_classes, activation='softmax')

---

### ⚙️ Training Configuration
| Component | Setting |
|------------|----------|
| **Regularization** | L2 (weight decay) |
| **Optimizer** | Adam with CosineDecayRestarts |
| **Loss** | Categorical Crossentropy (label smoothing = 0.1) |
| **Metrics** | Accuracy, Hierarchical F1 (custom metric) |

---

This architecture enables **multimodal temporal feature fusion** — combining local CNN feature extraction, long-range temporal modeling (via Bi-LSTM & Bi-GRU), and a learnable attention mechanism to emphasize key motion intervals during gesture execution.

📈 Experimental Results
Metric	Value
Binary F1 (BFRB/non-BFRB)	0.89
Macro F1 (18-class)	0.86
Average F1	0.875
Outperforms single-modal baselines (IMU-only: 0.75 Average F1) by 12 percentage points
SE attention and Mixup augmentation contribute ~4% performance gain each
Robust to sensor noise and participant variability

🎯 Future Improvements
Add real-time inference support for wearable devices
Extend to more BFRB subtypes and edge cases
Optimize model size for low-power embedded systems

## Requirements

- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Pandas
- Polars
- scikit-learn
- scipy


