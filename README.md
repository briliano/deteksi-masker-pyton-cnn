# 😷 Mask Detection CNN

Project deteksi masker wajah menggunakan metode **Convolutional Neural Network (CNN)** dengan TensorFlow, OpenCV, dan Keras.  
Sistem dapat melakukan training model serta mendeteksi penggunaan masker secara realtime melalui webcam.

---

# 📌 Features

- Training model CNN untuk klasifikasi masker
- Realtime mask detection menggunakan webcam
- Deteksi wajah menggunakan Haar Cascade
- Menampilkan label:
  - Mask
  - No Mask
- Menampilkan confidence prediction
- Save model `.h5`
- Visualisasi accuracy dan loss training

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Imutils

---

# 📂 Project Structure

```plaintext
deteksi-masker/
│
├── dataset/
│   └── train/
│       ├── with_mask/
│       └── without_mask/
│
├── models/
│
├── train_mask_detector.py
├── detect_mask_video.py
├── accuracy.png
├── loss.png
└── README.md
```

---

# ⚙️ Installation Guide

## 1. Install Python

Download dan install Python versi **3.10.11**:

https://www.python.org/downloads/release/python-31011/

Saat instalasi:
- Centang ✅ `Add Python to PATH`

Cek versi Python:

```bash
python --version
```

Output:

```bash
Python 3.10.11
```

---

## 2. Masuk ke Folder Project

```bash
cd D:\semester 6\computer vision\deteksi-masker
```

Sesuaikan dengan lokasi project masing-masing.

---

## 3. Buat Virtual Environment

```bash
python -m venv .venv
```

---

## 4. Aktifkan Virtual Environment

### CMD

```bash
.venv\Scripts\activate
```

Jika berhasil:

```bash
(.venv)
```

---

## 5. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 6. Install Dependencies

Install semua library yang dibutuhkan:

```bash
pip install tensorflow==2.10.1 opencv-python==4.9.0.80 imutils==0.5.4 matplotlib==3.8.4 scikit-learn==1.4.2 numpy==1.26.4
```

---

# 📦 Library Version

| Library | Version |
|---|---|
| Python | 3.10.11 |
| TensorFlow | 2.10.1 |
| NumPy | 1.26.4 |
| OpenCV | 4.9.0.80 |
| Matplotlib | 3.8.4 |
| Scikit-learn | 1.4.2 |
| Imutils | 0.5.4 |

---

# 🧠 Training Model CNN

Jalankan training model:

```bash
python train_mask_detector.py
```

Output:
- Training accuracy
- Validation accuracy
- Accuracy graph
- Loss graph
- Model `.h5`

Model akan tersimpan di:

```plaintext
models/mask_detector.h5
```

---

# 🎥 Realtime Mask Detection

Jalankan deteksi masker realtime:

```bash
python detect_mask_video.py
```

Fitur:
- Webcam realtime
- Deteksi wajah otomatis
- Prediksi Mask / No Mask
- Confidence score

Tekan tombol:

```plaintext
q
```

untuk keluar dari webcam.

---

# 📊 Result Visualization

Project akan menghasilkan:
- `accuracy.png`
- `loss.png`

Untuk melihat performa training model CNN.

---

# ❗ Troubleshooting

Jika TensorFlow error saat instalasi, install:

Microsoft Visual C++ Redistributable

https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

Download:
- `vc_redist.x64.exe`

Kemudian restart laptop.

---

# 🧑‍💻 Author

Developed by Brili Ano

---
