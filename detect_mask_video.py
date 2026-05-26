# Ubah baris ini:
# from tensorflow.keras.models import load_model

# Menjadi seperti ini:
from tensorflow.keras.models import load_model 
# ATAU jika masih error setelah instalasi:
from tensorflow import keras
# lalu panggil dengan: keras.models.load_model("models/mask_detector.h5")
import numpy as np
import cv2
import imutils

# load model cnn
model = load_model("models/mask_detector.h5")

# load haarcascade wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# buka webcam
cap = cv2.VideoCapture(0)

while True:

    # membaca frame webcam
    ret, frame = cap.read()

    # resize frame agar ringan
    frame = imutils.resize(frame, width=600)

    # convert ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60,60)
    )

    # looping semua wajah
    for (x,y,w,h) in faces:

        # crop wajah
        face = frame[y:y+h, x:x+w]

        try:
            # resize sesuai training cnn
            face = cv2.resize(face, (150,150))

            # normalisasi
            face = face / 255.0

            # ubah ke array numpy
            face = np.array(face)

            # tambah dimensi batch
            face = np.expand_dims(face, axis=0)

            # prediksi cnn
            pred = model.predict(face, verbose=0)[0]

            # ambil probabilitas
            mask = pred[0]
            no_mask = pred[1]

            # menentukan label
            if mask > no_mask:
                label = f"Mask {mask*100:.2f}%"
                color = (0,255,0)
            else:
                label = f"No Mask {no_mask*100:.2f}%"
                color = (0,0,255)

            # gambar kotak wajah
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

            # tampilkan teks
            cv2.putText(
                frame,
                label,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        except:
            pass

    # tampilkan webcam
    cv2.imshow("Deteksi Masker CNN", frame)

    # tekan q untuk keluar
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# tutup semua
cap.release()
cv2.destroyAllWindows()