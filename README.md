# 👆 AirClicker

This project explores finger detection for controlling a virtual mouse using just a single finger.
Built with TensorFlow, OpenCV, and simple grayscale processing,
it demonstrates how minimal input and lightweight modeling can deliver effective results in real-time interaction.

---

## 🚀 Features

- 🧠 Trained custom model with `.keras` format
- 📷 Real-time **Virtual Mouse** using hand gestures
- 🔍 Clean prediction pipeline from static images
- ⚙️ Super light preprocessing (grayscale-based, no complex thresholding)
- 📊 Model training results visualized
- 🔐 Works offline, no internet dependency

---

## 🔍 Going Beyond MediaPipe

While this virtual mouse could have been built entirely with MediaPipe, I chose to incorporate TensorFlow and model training to take it a step further.
A custom binary classifier was implemented to distinguish between:

- Click Gesture → One finger visible (Class 1)
- No Click → Any other gesture or hand configuration (Class 0)

## 📂 Project Structure

```
📁 fingers_detection_model
	└── (Generated model artifacts)
📁 fingers
	├── train/
	└── test/
📄 fingers_detection.keras       ← Final trained model
📄 model_training.ipynb          ← Notebook to train model
📄 testing.ipynb                 ← Testing model on static images
📄 virtual_mouse.py              ← Real-time finger detection & control
📄 training_metrics.png          ← Accuracy/Loss graphs
```

---

## 🧪 Try It Yourself

### 🔧 Requirements

- Python 3.10
- OpenCV
- TensorFlow
- NumPy

Install them via:

```bash
pip install tensorflow opencv-python numpy
```

---

### 🖼️ Test Static Images

Run this to see magic on sample images:

```bash
python testing.py
```

(Each image opens one at a time — press any key to continue.)

---

### 🖱️ Run Virtual Mouse (1-Finger Detection)

```bash
python virtual_mouse.py
```

Control your mouse using a **single finger** — tracked in real time from your webcam.

---

## 📈 Model Performance

![Training Metrics](model_metrices/training_metrics.png)

- Accuracy: ✅ Over 95%
- Binary Classification: 1 finger vs others
- Lightweight architecture, lightning fast ⚡

---

## 🙏 Why This Matters

In a world of overengineered solutions,  
this project is a **reminder** that simplicity wins.  
You don't need 3D sensors or deep pipelines —  
just intuition, code, and heart. ❤️

---

## 💡 Inspired By

- The joy of using hands to express
- The elegance of grayscale
- The belief that simple tools can feel like magic

---

## 📬 Contact

> Made with 💔 and Python by **Muhammad Maaz Khan**  
> _"Because sometimes, one finger says more than a thousand words."_

---

## 🌟 Star This Project

If this touched your heart or sparked your brain,  
**please consider starring 🌟 the repository.**

Let’s make minimalistic AI shine together ✨
