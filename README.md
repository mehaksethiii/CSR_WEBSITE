# CSR Detection from OCT Scan

A web app to detect **Central Serous Retinopathy (CSR)** from OCT (Optical Coherence Tomography) eye scan images using a deep learning model.

🔗 **Live Demo:** [csrwebsite-mehaksethiii.streamlit.app](https://csrwebsite-mehaksethiii.streamlit.app)

---

## What it does

Upload an OCT eye scan image and the app will tell you:
- ✅ **Normal Eye** — no CSR detected
- 🔴 **CSR Detected** — signs of Central Serous Retinopathy found

---

## Model

- Architecture: **MobileNetV2** (transfer learning)
- Trained on: 209 normal + 105 CSR OCT images
- Accuracy: **98%**
- Format: ONNX (for fast, lightweight inference)

---

## Tech Stack

- Python
- Streamlit
- ONNX Runtime
- Pillow / NumPy

---

## Run Locally

```bash
git clone https://github.com/mehaksethiii/CSR_WEBSITE.git
cd CSR_WEBSITE
pip install -r requirements.txt
streamlit run app.py
```

---

Made by [mehaksethiii](https://github.com/mehaksethiii)
