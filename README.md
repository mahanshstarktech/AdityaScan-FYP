# AdityaScan: Physics-Aware Solar Flare & Space Weather Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time multi-horizon solar flare forecasting combining ISRO Aditya-L1 satellite payload observations with physics-informed deep neural architectures.

---

## 👥 Project Team & Role Allocation

| Member | Primary Role | Specialization & Ownership |
| :--- | :--- | :--- |
| **Mahansh Gaur** *(Project Lead)* | **Lead ML Architect & MLOps** | Neural architectures (`backend/pipeline/ml/`), training loops (`notebooks/`), ONNX serving (`models/`), MLOps (`run_training.sh`) |
| **Sanskriti Raj** | **ML Scientist & Data Lead** | Satellite ingestion (`backend/pipeline/ingestion/`), physics proxies (`backend/pipeline/physics/`), uncertainty calibration (`uncertainty.py`) |
| **Somya Roy** | **Full-Stack Application & Operations Lead** | API service (`backend/pipeline/api/`), dashboard UI (`frontend/`), automated tests (`tests/`), deployment (`infra/`) |

---

## 🛰️ Architecture Overview

AdityaScan ingests solar observations from ISRO Aditya-L1 (SoLEXS, HEL1OS, MAG, ASPEX) and NOAA GOES XRS, extracting physics-guided features (Neupert effect derivative, Continuous Wavelet Transform) and processing them through a multi-modal fusion deep neural network with calibrated uncertainty.
