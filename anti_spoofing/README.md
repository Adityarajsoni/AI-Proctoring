# Anti-Spoofing Module

A lightweight and production-ready Face Anti-Spoofing module built using **MiniFASNetV2** and **RetinaFace (UniFace)**.

This module is part of a larger AI Proctoring System and is designed to detect presentation attacks such as:

* Printed photo attacks
* Mobile replay attacks
* Screen replay attacks
* Non-live face presentations

---

## Features

* Real-time webcam inference
* MiniFASNetV2 based liveness detection
* RetinaFace face detection
* Lightweight CPU-friendly architecture
* Modular service-based design
* Ready for FastAPI integration
* Suitable for online examination and identity verification systems

---

## Project Structure

```text
anti_spoofing/
│
├── models/
│   ├── fasnet.py
│   ├── model_loader.py
│   └── MiniFASNetV2.pth
│
├── services/
│   └── anti_spoof_service.py
│
├── utils/
│   ├── face_cropper.py
│   └── image_preprocess.py
│
├── test_webcam.py
├── config.py
└── requirements.txt
```

---

## Model

### Face Detection

* RetinaFace (via UniFace)

### Face Anti-Spoofing

* MiniFASNetV2
* Input Size: 80×80
* Lightweight architecture (~0.43M parameters)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Adityarajsoni/Anti-Spoofing.git
cd Anti-Spoofing
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Weights

Download the MiniFASNetV2 weights and place them inside:

```text
models/
└── MiniFASNetV2.pth
```

---

## Running Webcam Demo

```bash
python test_webcam.py
```

Press:

```text
q
```

to exit.

---

## Example Output

```python
{
    "label": "LIVE",
    "confidence": 0.997,
    "is_live": True
}
```

Replay attack:

```python
{
    "label": "SPOOF",
    "confidence": 0.032,
    "is_live": False
}
```

---

## Use Cases

* Online Proctoring
* Identity Verification
* Remote Examination Platforms
* KYC Systems
* Employee Authentication
* Attendance Monitoring

---

## Performance

Validated against:

* Real face input
* Mobile replay attacks

The model shows strong separation between genuine and spoofed presentations while remaining lightweight enough for real-time CPU inference.

---

## Future Improvements

* Active liveness detection
* Blink detection
* Head pose verification
* ONNX export support
* FastAPI integration
* Multi-face support

---

## Acknowledgements

This implementation is based on the MiniFASNet architecture and inspired by the Silent Face Anti-Spoofing project.

Original reference:

https://github.com/yakhyo/face-anti-spoofing

---

## License

This project is intended for educational, research, and commercial integration purposes. Please review the licenses of any third-party models and dependencies before production deployment.
