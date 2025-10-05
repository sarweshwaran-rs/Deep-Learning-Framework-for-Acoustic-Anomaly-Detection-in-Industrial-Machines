# DFCA-Net: Dual Frequency Cross-Attention Network

## 🔧 Industrial Machine Anomaly Detection System

A comprehensive deep learning solution for detecting anomalies in industrial machinery using audio signal analysis. The system combines STFT and CQT spectrograms through a novel Dual Frequency Cross-Attention Network (DFCA-Net) architecture.

![DFCA-Net Architecture](https://img.shields.io/badge/Architecture-DFCA--Net-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal) ![React](https://img.shields.io/badge/React-18+-blue)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Training](#model-training)
- [Performance](#performance)
---

## 🎯 Overview

DFCA-Net is an advanced anomaly detection system specifically designed for industrial machinery monitoring. By analyzing audio signals from machines, the system can detect potential faults and anomalies before they lead to costly failures.

### Key Capabilities:
- **Real-time Anomaly Detection**: Instant analysis of machine audio
- **High Accuracy**: Advanced deep learning with cross-attention mechanisms
- **Dual Frequency Analysis**: Combines STFT and CQT spectrograms
- **Web Interface**: User-friendly React frontend
- **REST API**: Comprehensive FastAPI backend

---

## Architecture

### DFCA-Net Model Architecture

The Dual Frequency Cross-Attention Network processes audio through multiple stages:

```
Audio Input → Preprocessing → Dual Frequency Extraction → Cross-Attention Fusion → Classification
     ↓              ↓                    ↓                        ↓                ↓
  .wav file    Resampling         STFT + CQT              Feature Fusion      Normal/Abnormal
               16kHz           Spectrograms              256-dimensional        + Confidence
```

### System Components:

1. **Audio Preprocessing**: Resampling, pre-emphasis filtering
2. **STFT Branch**: Short-Time Fourier Transform with mel-scale filtering
3. **CQT Branch**: Constant-Q Transform for harmonic analysis
4. **Cross-Attention Fusion**: Combines dual frequency representations
5. **Classification Head**: Binary anomaly detection with confidence scoring

### Technical Specifications:

| Component | Specification |
|-----------|---------------|
| **STFT FFT Size** | 512 |
| **STFT Hop Length** | 256 |
| **Mel Bands** | 64 |
| **CQT Bins** | 84 |
| **Bins per Octave** | 36 |
| **Dual Frequency Fusion Dimensions** | 256 |
| **Sample Rate** | 16kHz |
| **Detection Threshold** | 0.65 |

---

## ✨ Features

### Machine Learning
- **Dual Frequency Processing**: STFT + CQT spectrograms
- **Cross-Attention Mechanism**: Advanced feature fusion
- **Temporal Modeling**: Optional GRU-based temporal decoder
- **Confidence Scoring**: Probability-based predictions
- **GPU Acceleration**: CUDA support for faster inference

---

## 📁 Project Structure

```
DFCA/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── 
├── models/                           # Model architecture definitions
│   ├── __init__.py
│   ├── backbone.py                   # CNN backbone networks
│   ├── fusion.py                     # Cross-attention fusion modules
│   ├── heads.py                      # Classification heads
│   └── temporal.py                   # Temporal modeling components
│
├── scripts/                          # Training and utility scripts
│   ├── train.py                      # Main training script
│   ├── pretrain_pipeline.py          # Model pipeline definition
│   ├── data_loader.py                # Data loading utilities
│   └── utils.py                      # Helper functions
│
├── checkpoints/                      # Model weights and checkpoints
│   └── DFCAFinalNet/
│       └── best_model.pth           # Trained model weights
│
├── Notebooks/                        # Jupyter notebooks
│   ├── train.ipynb                  # Training experiments
│   ├── Final_Test_2.ipynb           # Model evaluation
│   └── cafm-model.ipynb             # Architecture development
│
├── dfca-net-api/                     # Backend API
│   ├── README.md                     # API documentation
│   ├── requirements.txt              # API dependencies
│   ├── run_api.py                    # API startup script
│   └── app/
│       ├── __init__.py
│       ├── main.py                   # FastAPI application
│       ├── predictor.py              # Model inference
│       ├── audio_processor.py        # Audio preprocessing
│       └── static/                   # Static files
│
├── frontend/                         # React web application
│   ├── README.md                     # Frontend documentation
│   ├── package.json                  # Node.js dependencies
│   ├── vite.config.js                # Vite configuration
│   ├── tailwind.config.js            # Tailwind CSS config
│   └── src/
│       ├── App.jsx                   # Main application
│       ├── main.jsx                  # Entry point
│       ├── index.css                 # Global styles
│       └── components/               # React components
│           ├── Navbar.jsx
│           ├── Hero.jsx
│           ├── About.jsx
│           ├── ApiInfo.jsx
│           ├── Predict.jsx
│           ├── ApiStatus.jsx
│           └── Footer.jsx
│
└── API_DOCUMENTATION.md              # Complete API reference
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** (for frontend)
- **CUDA** (optional, for GPU acceleration)
- **Git**

### 1. Backend Setup

```bash
# Navigate to API directory
cd dfca-net-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify model weights exist
ls checkpoints/DFCAFinalNet/best_model.pth
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install Node.js dependencies
npm install

---

## ⚡ Quick Start

### 1. Start Backend API

```bash
cd dfca-net-api
python run_api.py
```

### 2. Start Frontend (Development)

```bash
cd frontend
npm run dev
```

The web application will be available at:
- **Frontend**: http://localhost:5173
---

## 📖 Usage

### Web Interface

1. **Upload Audio**: Drag and drop or select a .wav file
2. **Preview**: Use play/pause/stop controls to listen
3. **Analyze**: Click "Detect Anomaly" for prediction
4. **Results**: View Normal/Abnormal classification


#### Python Example

```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Anomaly detection
files = {"file": ("machine.wav", open("machine.wav", "rb"), "audio/wav")}
response = requests.post("http://localhost:8000/predict/", files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidicence']}")
```
### File Requirements

- **Format**: .wav files only
- **Duration**: 1-10 seconds recommended
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **File Size**: Maximum 50MB (< 10MB recommended)
- **Content**: Clear machine audio without excessive background noise

---

### Model Architecture Details

- **Backbone**: ResNet-based CNN for feature extraction
- **Fusion**: Cross-Attention Fusion Module (CAFM)
- **Head**: Anomaly scoring with dropout regularization
- **Temporal**: Optional GRU-based temporal smoothing

---

## 📊 Performance

### Model Metrics

- **Accuracy**: >90% on test datasets
- **Precision**: High precision for anomaly detection
- **Recall**: Effective fault detection capability
- **F1-Score**: Balanced performance metrics

### System Performance

- **Inference Time**: 1-8 seconds depending on file size
- **Memory Usage**: ~2GB GPU memory for inference
- **Throughput**: Multiple concurrent requests supported
- **Scalability**: Horizontal scaling with load balancers

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8GB
- **Storage**: 5GB free space
- **GPU**: Optional (CPU inference supported)

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 16GB+
- **Storage**: 10GB+ SSD
- **GPU**: NVIDIA GTX 1060+ or equivalent

---

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
MODEL_PATH=checkpoints/DFCAFinalNet/best_model.pth
THRESHOLD=0.65
MAX_FILE_SIZE=50MB
LOG_LEVEL=INFO

# CORS Settings
ALLOWED_ORIGINS=http://localhost:5173

# GPU Settings
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

```python
# Model parameters
STFT_DIM = 512
CQT_DIM = 320
FUSION_DIM = 256
THRESHOLD = 0.65
SAMPLE_RATE = 16000
```

---

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```bash
   # Check model path and permissions
   ls -la checkpoints/DFCAFinalNet/best_model.pth
   ```

2. **CUDA/GPU Issues**
   ```bash
   # Verify PyTorch CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **File Upload Errors**
   - Ensure file is in .wav format
   - Check file size (< 50MB)
   - Verify audio file is not corrupted

4. **API Connection Issues**
   - Confirm backend is running on port 8000
   - Check CORS settings for frontend URL
   - Verify firewall settings

### Debug Mode

Enable detailed logging:

```bash
# Backend debug mode
LOG_LEVEL=DEBUG python run_api.py

# Frontend development mode
npm run dev
```
### Documentation
- **API Docs**: http://localhost:8000/docs


## 🙏 Acknowledgments

- Industrial machine audio datasets
- PyTorch and FastAPI communities
- React and Tailwind CSS frameworks
- Research contributions in anomaly detection

