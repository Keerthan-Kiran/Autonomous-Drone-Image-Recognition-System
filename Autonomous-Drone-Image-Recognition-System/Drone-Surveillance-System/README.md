# 🚁 YOLOv8 Drone Surveillance System

![YOLOv8](https://img.shields.io/badge/YOLO-v8-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

A professional computer vision project for aerial surveillance. This system uses **YOLOv8 Medium** for high-accuracy detection and an ML-based anomaly detector to lock onto targets of interest.

## ✨ Key Features
* **High-Accuracy Detection:** Uses `yolov8m.pt` (Medium) running at High-Res (1280px).
* **Smart Locking:**
    * **Size Locking:** Targets large vehicles automatically.
    * **Anomaly Locking:** Uses `IsolationForest` to detect irregular behavior.
* **Modular Structure:** Clean separation of logic, config, and execution.

## 📂 Structure
