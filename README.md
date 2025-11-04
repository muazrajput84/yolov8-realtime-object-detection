##🎯 YOLOv8 Real-Time Object Detection

Author: M. Muaz  
Role:   AI/ML Engineer  
Environment: Conda (Python 3.10)  
Frameworks: PyTorch, OpenCV, Ultralytics, NumPy  

##🧠 Overview

This project demonstrates a **real-time object detection system** built using **YOLOv8 (Ultralytics)** — a state-of-the-art object detection architecture.  
The system can identify and label multiple objects directly from your **webcam feed** with high speed and accuracy.

This project is ideal for portfolios showcasing:
- Computer Vision skills  
- Deep Learning and PyTorch knowledge  
- Real-time image processing and AI pipeline development  

## 🚀 Key Features

✅ Real-time object detection using webcam  
✅ Bounding boxes with class labels and confidence scores  
✅ Screenshot capture of detections (`S` key)  
✅ Optimized for both CPU and GPU environments  
✅ Modular and easy to extend for custom datasets  
✅ Fully open-source and free to use  

## 🖼️ Detectable Objects

The YOLOv8 model (trained on **COCO dataset**) detects **80+ object categories**, including:
- 👨 Person  
- 🚗 Car  
- 🚲 Bicycle  
- 🐕 Dog  
- 🐈 Cat  
- 🧳 Backpack  
- 🚦 Traffic light  
- 🏠 Chair, TV, Laptop, etc.  

# ⚙️ Installation Guide — YOLOv8 Real-Time Object Detection

This guide will help you set up and run the **YOLOv8 Real-Time Object Detection** project on your local machine using **Anaconda (Conda)** or **Google Colab**.

## 🧰 System Requirements

Recommended:
- OS: Windows 10/11, macOS, or Linux  
- Python: 3.10  
- GPU: Optional (CUDA 11.8+ for faster inference)  
- RAM: 4GB minimum (8GB recommended)  

## 🪜 Step-by-Step Installation (Local — Conda)

### 1️⃣ Install Anaconda or Miniconda
If you haven’t already, download and install **Miniconda** or **Anaconda**:
- [Miniconda (recommended)](https://docs.conda.io/en/latest/miniconda.html)  
- [Anaconda](https://www.anaconda.com/products/distribution)

After installation, open Anaconda Prompt.

---

### 2️⃣ Clone the Project Repository
Run this command in the Anaconda prompt:

```bash
cd C:\Users\<YourUserName>\Downloads
git clone https://github.com/<your-username>/yolov8-realtime-object-detection.git
cd yolov8-realtime-object-detection
````

---

### 3️⃣ Create a Conda Environment

Create a new isolated environment for the project:

```bash
conda create -n yolov8env python=3.10 -y
```

Activate the environment:

```bash
conda activate yolov8env
```

---

### 4️⃣ Install Dependencies

Install all required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Example of `requirements.txt`:

```txt
ultralytics==8.0.200
torch==2.3.0
opencv-python==4.10.0.84
numpy==1.26.4
```

> 💡 If you face issues installing `torch`, visit [PyTorch.org](https://pytorch.org/get-started/locally/) for GPU-specific install commands.


### 5️⃣ Download YOLOv8 Model

The script automatically downloads the **YOLOv8n** model on first run:

```python
model = YOLO('yolov8n.pt')
```

If you prefer a more accurate model, you can manually download one:

* [YOLOv8 Models (Ultralytics)](https://github.com/ultralytics/ultralytics)

Available variants:

| Model      | Speed     | Accuracy  | Recommended Use      |
| ---------- | --------- | --------- | -------------------- |
| yolov8n.pt | ⚡ Fastest | Moderate  | CPU or low-end GPU   |
| yolov8s.pt | ⚡         | Good      | Real-time detection  |
| yolov8m.pt | ⚙️        | Better    | Balanced accuracy    |
| yolov8l.pt | 🧠        | High      | Research or analysis |
| yolov8x.pt | 🔥        | Very High | Heavy GPU            |

---

### 6️⃣ Run the Project

Now simply run:

```bash
python main.py
```

This opens your webcam and starts object detection.

Controls:

Press Q → Quit the program
Press S → Save the current frame (saved in `screenshots/`)

---

## 🧠 Alternate Setup — Google Colab (Optional)

If you don’t want to run locally or lack GPU access:

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload all project files (`main.py`, `requirements.txt`, etc.)
3. Run these cells:

```python
!pip install -r requirements.txt
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.predict(source='https://ultralytics.com/images/bus.jpg', show=True)
```

4. Colab doesn’t support webcam feed directly — use **video files** or sample images instead.

---

## 🧩 Folder Structure

```
yolov8-realtime-object-detection/
│
├── main.py                  # Main script (object detection)
├── requirements.txt         # Dependencies list
├── README.md                # Documentation
├── INSTALLATION_GUIDE.md    # (This file)
│
├── models/                  # Pretrained YOLO models (optional)
├── screenshots/             # Saved images
└── utils/                   # Helper functions (optional)
```

---

## 🔍 Common Issues & Fixes

| Issue                | Cause                           | Fix                                                                                         |
| -------------------- | ------------------------------- | ------------------------------------------------------------------------------------------- |
| `torch not found`    | PyTorch not installed correctly | Reinstall with correct command from [pytorch.org](https://pytorch.org/get-started/locally/) |
| `cv2 not found`      | OpenCV missing                  | Run `pip install opencv-python`                                                             |
| `Camera not opening` | Permission or driver issue      | Ensure your webcam is not used by another app                                               |
| `Slow performance`   | Using CPU                       | Enable GPU (CUDA) if available                                                              |

---

## ⚡ GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To verify GPU:

```python
import torch
print(torch.cuda.is_available())
```

If `True`, YOLOv8 will automatically use GPU for inference.

---

## 🧩 Uninstallation (Optional)

To remove the environment:

```bash
conda deactivate
conda remove -n yolov8env --all -y
```

---

## ✅ You’re Ready!

You now have a fully working real-time object detection system using YOLOv8.

If everything works fine:

You’ll see bounding boxes over detected objects.
FPS and confidence scores will display in real-time. 
Screenshots will save automatically in `/screenshots`.


## 🧑‍💻 Author

M. Muaz
AI/ML Engineer | Computer Vision | Deep Learning Enthusiast
📧 Email: muazrajput84@gmail.com
🌐 LinkedIn: https://linkedin.com/in/muazrajput84
🐙 GitHub: https://github.com/muazrajput84
