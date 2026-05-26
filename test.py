import cv2

faceCascade = cv2.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_default.xml'
)

print("Haar Cascade berhasil dibaca")