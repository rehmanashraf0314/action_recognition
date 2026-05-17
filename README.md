# 🏋️ Exercise Recognition using MediaPipe Pose + GRU

<div align="center">

## Human Activity Recognition from Skeletal Motion

Lightweight deep learning pipeline for exercise recognition using:

`MediaPipe Pose` • `Signal Processing` • `Feature Engineering` • `GRU Networks`

</div>

---

# 📌 Overview

This project implements a complete **Human Activity Recognition (HAR)** pipeline for recognizing gym exercises from videos using:

- MediaPipe Pose Landmarker
- Pose World Landmarks
- Rep Segmentation
- Signal Processing
- Pose Normalization
- Angle-based Feature Engineering
- Temporal Sequence Modeling
- GRU Deep Learning Network

Instead of training directly on RGB video frames, this system learns:

> 🧠 **Human body movement patterns through skeletal motion sequences**

---

# 🚀 Supported Exercises

| Exercise |
|---|
| Push-up |
| Squat |
| Barbell Biceps Curl |
| Shoulder Press |

---

# 🧠 Why Pose-Based Recognition?

Traditional video-based action recognition models:

- require massive datasets
- are computationally expensive
- depend heavily on backgrounds and lighting
- need GPUs for efficient training

This project instead uses:

```text
Human skeletal motion
```

Advantages:

✅ Lightweight  
✅ CPU-friendly  
✅ Faster training  
✅ Better motion understanding  
✅ Background invariant  
✅ Easier temporal learning  

---

# 🔥 Complete Pipeline

```text
Video
 ↓
MediaPipe Pose Extraction
 ↓
Pose World Landmarks
 ↓
Signal Generation
 ↓
Peak/Valley Detection
 ↓
Rep Segmentation
 ↓
Pose Normalization
 ↓
Feature Engineering
 ↓
Temporal Normalization
 ↓
GRU Training
 ↓
Exercise Classification
```

---

# 📹 MediaPipe Pose Extraction

The system uses:

```python
MediaPipe Pose Landmarker (VIDEO mode)
```

For every frame:

- 33 body landmarks are extracted
- 3D world coordinates are obtained
- Temporal consistency is maintained

Each landmark contains:

```python
x, y, z, visibility
```

---

# 🌍 Why Pose World Landmarks?

The project uses:

```python
pose_world_landmarks
```

instead of normalized image coordinates.

### Why?

Normalized image coordinates depend on:

- camera distance
- image resolution
- frame position
- subject scale

World landmarks provide:

✅ Better spatial consistency  
✅ Better motion representation  
✅ More stable geometry  
✅ Better temporal learning  

---

# 📈 Rep Detection using Signal Processing

A motion signal is generated from body movement.

Examples:

| Exercise | Signal |
|---|---|
| Push-up | Shoulder-Wrist distance |
| Squat | Hip-Knee distance |
| Bicep Curl | Shoulder-Wrist distance |
| Shoulder Press | Wrist vertical movement |

The signal is:

1. Smoothed
2. Normalized
3. Processed using peak/valley detection

Using:

```python
scipy.signal.find_peaks
```

---

# 🔄 Rep Segmentation

A single exercise video can contain:

- multiple repetitions
- different speeds
- inconsistent frame counts

Instead of training on full videos:

✅ Each repetition becomes one training sample.

This creates:

```text
One Rep = One Sequence Sample
```

---

# 🎯 Exact Rep Segmentation Logic

The system extracts exact repetition boundaries using:

```text
Previous Peak → Valley → Next Peak
```

This allows the extraction of:

✅ Complete motion cycles  
✅ Clean temporal samples  
✅ Better GRU learning  

---

# 🧠 Feature Engineering

Each frame is converted into a feature vector.

The system combines:

---

## 1️⃣ Normalized 3D Landmarks

33 landmarks × 3 coordinates:

```text
99 features
```

---

## 2️⃣ Joint Angles

Important biomechanical angles:

- Left elbow
- Right elbow
- Left shoulder
- Right shoulder
- Left knee
- Right knee
- Left hip
- Right hip

Total:

```text
8 angle features
```

---

# 💡 Why Angles?

Angles are:

✅ Scale invariant  
✅ Biomechanically meaningful  
✅ Robust to body size changes  

Example:

```text
90° elbow angle
```

represents the same motion for both tall and short individuals.

---

# 📏 Pose Normalization

Two major normalization steps are applied.

---

## 🔹 Translation Normalization

Body center is moved to origin.

Using:

```text
Hip Center
```

This removes:

- camera offset
- body position variation

---

## 🔹 Scale Normalization

Landmarks are divided using:

- shoulder width
- torso size

This removes:

- person height variation
- distance from camera

---

# ⏱ Temporal Normalization

Different reps contain different frame counts.

Examples:

| Rep Type | Frames |
|---|---|
| Fast Rep | 25 |
| Slow Rep | 80 |

GRU requires fixed-length sequences.

Therefore:

```text
Every repetition is interpolated to 40 frames
```

using sequence interpolation.

---

# 🗂 Dataset Structure

The dataset is automatically generated.

```text
my_dataset/
│
├── pushup/
│   ├── pushup_0.npz
│   ├── pushup_1.npz
│
├── squat/
│   ├── squat_0.npz
│   ├── squat_1.npz
│
├── shoulder_press/
│
├── bicep_curl/
```

Each `.npz` file contains:

```python
features
label
```

Feature shape:

```text
(40, 107)
```

Where:

| Dimension | Meaning |
|---|---|
| 40 | normalized frames |
| 107 | feature dimension |

---

# 🧬 Why Store Reps Individually?

Advantages:

✅ Modular dataset  
✅ Easy debugging  
✅ Easy augmentation  
✅ Cleaner training samples  
✅ Better scalability  

---

# 🤖 GRU Network

The project uses a:

```text
GRU-based sequence classifier
```

Architecture:

```text
Input Sequence
 ↓
GRU(128)
 ↓
GRU(64)
 ↓
Dense Layer
 ↓
Softmax Classification
```

---

# ⚡ Why GRU?

GRU is ideal for:

- temporal sequence learning
- motion understanding
- lightweight recurrent modeling

Advantages over LSTM:

✅ Faster training  
✅ Fewer parameters  
✅ Better for smaller datasets  

---

# 📦 GRU Input Format

The model receives:

```text
(samples, frames, features)
```

Example:

```text
(820, 40, 107)
```

Meaning:

| Dimension | Meaning |
|---|---|
| 820 | repetitions |
| 40 | normalized frames |
| 107 | features/frame |

---

# 🏷 Label Encoding

Exercise labels are encoded using:

```python
LabelEncoder
```

Example:

| Index | Exercise |
|---|---|
| 0 | bicep_curl |
| 1 | pushup |
| 2 | shoulder_press |
| 3 | squat |

---

# 📊 Training Results

The GRU achieved high validation accuracy using:

✅ Pose-based features  
✅ Rep segmentation  
✅ Motion normalization  
✅ Temporal alignment  

The model learns:

```text
Body movement patterns
```

instead of image appearance.

---

# 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| MediaPipe | Pose extraction |
| OpenCV | Video processing |
| NumPy | Numerical computation |
| SciPy | Signal processing |
| TensorFlow / Keras | Deep learning |
| scikit-learn | Data preparation |
| Matplotlib | Visualization |

---

# 🔮 Future Improvements

Possible extensions:

- Real-time webcam inference
- Real-time rep counting
- Exercise form correction
- Automatic exercise segmentation
- Transformer-based sequence models
- Multi-person exercise recognition
- Real-time coaching system
- Rep quality scoring

---

# 🎓 Key Learnings

This project demonstrates:

✅ Human Activity Recognition  
✅ Pose-based motion analysis  
✅ Signal processing for rep detection  
✅ Temporal sequence learning  
✅ Feature engineering for biomechanics  
✅ GRU-based action recognition  

---

# 🏁 Conclusion

This project builds a complete end-to-end exercise recognition pipeline using:

- Pose Estimation
- Rep Segmentation
- Pose Normalization
- Feature Engineering
- Temporal Alignment
- GRU-based Classification

Instead of relying on heavy video models, the system learns:

> 🧠 Human motion directly from skeletal movement patterns.

This creates a lightweight, scalable, and effective Human Activity Recognition system for exercise understanding.

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a star!

</div>
