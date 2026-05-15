# 🌶 ChilliVision — Explainable AI Framework for Automated Chilli Quality Grading

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

A complete end-to-end computer vision system for identifying Indian red chilli varieties and grading their quality using deep learning with explainability (GradCAM). Powered by MobileNetV2 and enhanced with computer vision analytics.

**[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API Docs](#-api-reference) • [License](#-license)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Supported Varieties](#-supported-chilli-varieties)
- [API Reference](#-api-reference)
- [Training](#-training-your-own-model)
- [Tech Stack](#-tech-stack)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

ChilliVision is an intelligent quality assessment system designed specifically for Indian red chilli varieties. It combines deep learning classification with explainable AI (GradCAM) and computer vision analytics to provide:

- **Automated variety identification** across 5 major Indian chilli types
- **Quality grading** from poor to premium grades
- **Visual explainability** showing which image regions drive predictions
- **6-metric quality analysis** based on color, texture, and defects
- **RESTful API** for integration with external systems
- **User-friendly web interface** for easy image upload and analysis

---

## ✨ Features

| Feature                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| **Multi-Class Classification** | Identifies 5 distinct Indian chilli varieties     |
| **Quality Scoring**            | AI-powered quality assessment (0–100 scale)       |
| **Explainable AI (GradCAM)**   | Visual heatmaps explaining model decisions        |
| **Computer Vision Analytics**  | 6 specialized metrics for quality assessment      |
| **Mobile-Optimized**           | MobileNetV2 backbone for edge deployment          |
| **REST API**                   | Flask backend for easy integration                |
| **Web Interface**              | Responsive frontend for image upload and analysis |
| **Real-Time Processing**       | Sub-second inference time per image               |

---

## 📋 Project Structure

```
chilli_grading/
├── backend/
│   ├── app.py                  # Flask REST API server
│   └── chilli_net.pth          # Pre-trained model weights
├── frontend/
│   └── index.html              # Web UI (HTML + CSS + JavaScript)
├── model/
│   └── chilli_model.py         # ChilliNet architecture + GradCAM + CV analysis
├── data/
│   ├── my_chilli_images/       # Training data directory
│   └── sample_images/          # Example images for testing
├── utils/                       # Utility functions
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── chilli_net.pth              # Pre-trained model weights
├── start.sh                    # Unix/Linux startup script
└── start.bat                   # Windows startup script
```

---

## 📦 Requirements

- **Python** 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Disk Space**: ~500 MB (with model weights)
- **GPU** (optional): NVIDIA GPU with CUDA support for faster training

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Train with Your Own Data

Organize your chilli images like this:

```
data/my_chilli_images/
├── Byadgi/
│   ├── Byadgi Bulk/
│   └── Byadgi Single/
├── Kashmiri/
│   ├── Kashmiri Bulk/
│   └── Kashmiri Single/
├── Guntur/
│   ├── Guntur Bulk/
│   └── Guntur Single/
├── Lavangi/
│   ├── Lavangi Bulk/
│   └── Lavangi Single/
└── Sankeshwari/
    └── Sankeshwari Single/
```

Then run:

```bash
python train_model.py --data ./data/my_chilli_images --epochs 30 --batch 32 --lr 0.001
```

Or use the included pre-trained model (recommended):

```bash
python train_model.py --epochs 20
```

### 3. Start the Application

**Option A: Automated startup (recommended)**

Linux/macOS:

```bash
bash start.sh
```

Windows:

```cmd
start.bat
```

**Option B: Manual startup**

Terminal 1 - Start Backend:

```bash
cd backend
python app.py
```

The server runs at: **http://localhost:5000**

### 4. Access the Application

Open your web browser and navigate to:

```
http://localhost:5000
```

Or directly open `frontend/index.html` in your browser.

---

## 🧠 Architecture

### ChilliNet Model Architecture

**Neural Network Structure:**

- **Backbone**: MobileNetV2 (lightweight, optimized for mobile deployment)
- **Feature Extraction**: Pre-trained on ImageNet, fine-tuned for chilli varieties
- **Classification Head**: 1280 → 512 → 256 → 5 classes
  - Dense layer 1: 1280 → 512 neurons (ReLU activation)
  - Dropout: 0.5 (prevents overfitting)
  - Dense layer 2: 512 → 256 neurons (ReLU activation)
  - Dropout: 0.5
  - Output: 5-class softmax (variety classification)
- **Quality Regression Head**: 256 → 64 → 1
  - Estimates quality score (0–100)
  - Sigmoid activation for bounded output

**Dual Output System:**

- **Type Prediction**: Which chilli variety (5 classes)
- **Quality Score**: Predicted quality (0–100 scale)

### Explainability: GradCAM

Gradient-weighted Class Activation Mapping provides visual explanations:

- Generates class-specific heatmaps
- Shows which image regions influenced the classification
- Overlaid on original image for intuitive interpretation
- Helps identify focus on relevant features (color, defects, shape)

### Computer Vision Quality Analysis

**6-Metric Comprehensive Assessment:**

| Metric                 | Description                                       | Weight | Range |
| ---------------------- | ------------------------------------------------- | ------ | ----- |
| **Red Dominance**      | Red channel intensity relative to G/B channels    | High   | 0–100 |
| **Colour Saturation**  | HSV saturation mean (vibrancy)                    | High   | 0–100 |
| **Surface Uniformity** | Inverse std of grayscale intensities (smoothness) | Medium | 0–100 |
| **Edge Sharpness**     | Laplacian variance (texture clarity)              | Medium | 0–100 |
| **Defect-Free Score**  | Absence of dark spots, rot, or blemishes          | High   | 0–100 |
| **Texture Quality**    | Consistency of surface texture patches            | Medium | 0–100 |

### Quality Grading Scale

| Grade       | Score Range | Label           | Description                                          |
| ----------- | ----------- | --------------- | ---------------------------------------------------- |
| **Grade A** | 85–100      | Premium Quality | Excellent appearance, minimal defects, vibrant color |
| **Grade B** | 65–84       | Good Quality    | Good appearance, minor imperfections                 |
| **Grade C** | 40–64       | Average Quality | Average appearance, some visible defects             |
| **Grade D** | 0–39        | Poor Quality    | Significant defects, damaged or aged                 |

---

## 🌶 Supported Chilli Varieties

ChilliVision recognizes these major Indian chilli varieties:

| Variety           | Region                | Characteristics                                         | Scoville SHU  |
| ----------------- | --------------------- | ------------------------------------------------------- | ------------- |
| **Byadagi**       | Karnataka, India      | Large, dark red, dried for export, good color intensity | 10,000–50,000 |
| **Kashmiri**      | Kashmir, India        | Bright red, mild spice, primarily for color/powder      | 1,000–2,000   |
| **Guntur Sannam** | Andhra Pradesh, India | Long, slim, high pungency, hot and aromatic             | 35,000–40,000 |
| **Lavangi**       | Various regions       | Medium-sized, balanced heat and flavor                  | 15,000–30,000 |
| **Sankeshwari**   | Various regions       | High-quality, excellent color and spice                 | 20,000–40,000 |

---

## 🔌 API Reference

### Health Check Endpoint

**GET `/api/health`**

Health status and model information.

**Response:**

```json
{
  "status": "healthy",
  "model": "ChilliNet (MobileNetV2)",
  "version": "1.0",
  "supported_formats": ["jpg", "jpeg", "png"],
  "max_file_size_mb": 10
}
```

---

### Image Analysis Endpoint

**POST `/api/analyse`**

Submit an image for comprehensive chilli analysis.

**Request:**

- Content-Type: `multipart/form-data`
- Field: `image` (binary image file, JPEG/PNG)
- Supported formats: JPG, JPEG, PNG
- Max size: 10 MB

**Response:**

```json
{
  "status": "success",
  "predicted_type": "Kashmiri Chilli",
  "type_confidence": 78.4,
  "origin": "Kashmir, India",
  "scoville": "1,000–2,000 SHU",
  "quality_grade": "Grade A",
  "quality_score": 88.2,
  "quality_label": "Premium Quality",
  "is_good": true,
  "recommendation": "✅ Kashmiri Chilli is premium grade. Excellent for powder production.",
  "top3_predictions": [
    { "type": "Kashmiri Chilli", "confidence": 78.4 },
    { "type": "Byadagi Chilli", "confidence": 15.2 },
    { "type": "Guntur Sannam", "confidence": 5.8 }
  ],
  "cv_features": {
    "red_dominance": 92.3,
    "colour_saturation": 85.1,
    "surface_uniformity": 78.4,
    "edge_sharpness": 88.2,
    "defect_free_score": 90.1,
    "texture_quality": 82.5
  },
  "gradcam_image": "data:image/png;base64,...",
  "original_image": "data:image/jpeg;base64,...",
  "processing_time_s": 0.84,
  "timestamp": "2026-05-15T10:30:45Z"
}
```

**Error Response:**

```json
{
  "status": "error",
  "message": "Invalid image format or corrupted file",
  "code": "INVALID_IMAGE"
}
```

---

### Grades Information Endpoint

**GET `/api/grades`**

Retrieve all grade definitions and chilli information.

**Response:**

```json
{
  "grades": [...],
  "varieties": [...],
  "features": [...]
}
```

---

## 🎓 Training Your Own Model

### Data Collection Requirements

For best results, collect high-quality training data:

- **Minimum**: 50–100 images per variety
- **Recommended**: 200+ images per variety
- **Variations needed**:
  - Different lighting conditions (natural, artificial, indoor, outdoor)
  - Multiple backgrounds (white, colored, realistic)
  - Various distances and angles
  - Fresh and slightly aged samples

### Training Steps

1. **Organize your data:**

   ```
   data/my_chilli_images/
   ├── Byadgi/
   ├── Kashmiri/
   ├── Guntur/
   ├── Lavangi/
   └── Sankeshwari/
   ```

2. **Run training with custom parameters:**

   ```bash
   python train_model.py \
     --data ./data/my_chilli_images \
     --epochs 40 \
     --batch 32 \
     --lr 0.001 \
     --val_split 0.2
   ```

3. **Training arguments:**
   - `--data`: Path to training images directory
   - `--epochs`: Number of training epochs (default: 30)
   - `--batch`: Batch size (default: 32)
   - `--lr`: Learning rate (default: 0.001)
   - `--val_split`: Validation split ratio (default: 0.2)

4. **Monitor training:**
   - Training logs displayed in console
   - Model saved to `backend/chilli_net.pth` automatically
   - Best model checkpoint saved during training

### Tips for Best Results

- Use images at 224×224 resolution minimum
- Ensure good lighting and clear visibility
- Avoid blurry or overexposed images
- Include various defects (if grading quality)
- Use data augmentation for small datasets
- Validate on unseen test data

---

## 🛠 Tech Stack

| Component                   | Technology                      | Version               |
| --------------------------- | ------------------------------- | --------------------- |
| **Deep Learning Framework** | PyTorch                         | 1.9+                  |
| **Pre-trained Model**       | MobileNetV2                     | ImageNet              |
| **Explainability**          | GradCAM                         | Custom Implementation |
| **Computer Vision**         | OpenCV                          | 4.5+                  |
| **Image Processing**        | Pillow                          | 8.0+                  |
| **Image Transforms**        | torchvision                     | 0.10+                 |
| **Backend Framework**       | Flask                           | 2.0+                  |
| **CORS Support**            | Flask-CORS                      | 3.0+                  |
| **Frontend**                | HTML5, CSS3, Vanilla JavaScript | ES6+                  |
| **Numeric Computing**       | NumPy                           | 1.19+                 |

---

## 📊 Performance Metrics

### Model Performance

- **Accuracy**: ~92% on validation set
- **Inference Time**: ~0.8–1.2 seconds per image
- **Model Size**: ~13 MB (compressed)
- **Memory Usage**: ~200 MB (with dependencies)

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

---

## 🔧 Troubleshooting

### Issue: Model not found error

**Error:**

```
FileNotFoundError: chilli_net.pth not found
```

**Solution:**

1. Ensure `chilli_net.pth` exists in the `backend/` directory
2. Download pre-trained weights from the project repository
3. Or retrain the model: `python train_model.py`

---

### Issue: CUDA out of memory during training

**Error:**

```
RuntimeError: CUDA out of memory
```

**Solution:**

1. Reduce batch size: `python train_model.py --batch 16`
2. Use CPU instead: Set `device='cpu'` in training script
3. Clear GPU cache between training runs

---

### Issue: Flask server not accessible

**Error:**

```
Connection refused on localhost:5000
```

**Solution:**

1. Verify backend is running: `python backend/app.py`
2. Check firewall settings
3. Try different port: Modify `app.run(port=5001)` in `backend/app.py`
4. Verify Flask installation: `pip list | grep Flask`

---

### Issue: Image upload failing

**Error:**

```
Invalid image format or corrupted file
```

**Solution:**

1. Ensure image is in JPG or PNG format
2. Check file size (max 10 MB)
3. Verify image is not corrupted: Open in image viewer first
4. Try re-exporting the image from an image editor

---

### Issue: Low accuracy in training

**Causes & Solutions:**

1. **Insufficient training data**: Collect more images (200+ per class)
2. **Class imbalance**: Ensure similar number of images per variety
3. **Poor image quality**: Use high-resolution, well-lit images
4. **Overfitting**: Increase dropout rate or reduce model complexity
5. **Learning rate too high**: Reduce to 0.0001
6. **More epochs needed**: Try `--epochs 50` or more

---

## 📝 Usage Examples

### Python Integration

```python
from model.chilli_model import ChilliNet
import torch
from PIL import Image

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChilliNet().to(device)
model.load_state_dict(torch.load('backend/chilli_net.pth'))
model.eval()

# Analyze image
image_path = 'data/sample_images/sample.jpg'
image = Image.open(image_path)

with torch.no_grad():
    result = model.analyze(image)
    print(f"Type: {result['predicted_type']}")
    print(f"Quality: {result['quality_grade']}")
    print(f"Confidence: {result['type_confidence']:.1f}%")
```

### cURL Integration

```bash
curl -X POST -F "image=@chilli_image.jpg" http://localhost:5000/api/analyse
```

---

## 👥 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Commit changes**: `git commit -m "Add your feature"`
4. **Push to branch**: `git push origin feature/your-feature`
5. **Submit Pull Request** with detailed description

---

## 📄 License

This project is licensed under the **MIT License** — free for academic and commercial use.

See [LICENSE](LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---




