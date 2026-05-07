# SeniorProject2-Group10

ElevAIte is an AI-powered smart elevator monitoring system designed to improve residential elevator safety using computer vision, deep learning, and Explainable AI (XAI).
The system detects critical abnormal events and provides immediate alerts with interpretable insights.

---

# Features

* Fainting Detection using CNN-BiLSTM
* Unsupervised Child Detection using YOLO + Machine Learning
* Abnormal Door Behavior Detection using BiLSTM
* Explainable AI (XAI) for interpretable event explanations
* Modular multi-worker architecture with combined decision-making system.
  
---

# System Architecture

The system is divided into independent modules:

| Module                   | Description                          |
| ------------------------ | ------------------------------------ |
| `faint_worker.py`        | Handles fainting detection           |
| `child_worker.py`        | Detects unattended children          |
| `door_worker.py`         | Detects abnormal door behavior       |
| `XAI_worker.py`          | Generates interpretable explanations |

---

# Technologies Used

* Python
* YOLO
* OpenCV
* TensorFlow / Keras
* Scikit-learn
* CNN-BiLSTM
* Optical Flow
* Explainable AI (XAI)

---

# Explainable AI (XAI)

The system integrates Explainable AI to improve transparency and trustworthiness.

Instead of providing only alerts, the system generates interpretable insights explaining:

* Why an event was detected
* Important motion patterns
* Key visual evidence contributing to the decision

---

# Research Contribution

ElevAIte contributes to intelligent elevator safety systems by combining:

* Deep learning
* Computer Vision
* Multi-event detection
* Explainable AI

to create a safer and more transparent smart elevator environment.

---

# License

This project is for research and educational purposes.
