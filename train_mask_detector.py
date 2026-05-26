# TRAIN MASK DETECTOR CNN

# import library untuk pengolahan gambar, data, dan CNN
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

# library split dataset
from sklearn.model_selection import train_test_split

# library one hot encoding
from tensorflow.keras.utils import to_categorical

# library model CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# LOAD DATASET

# menyiapkan variabel penyimpanan data gambar dan label
data, labels = [], []

# kategori dataset
categories = ["with_mask", "without_mask"]

# lokasi dataset
dataset_path = "dataset/train"

# membaca dataset gambar satu per satu
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        try:
            # membaca gambar menggunakan opencv
            image = cv2.imread(img_path)

            # resize gambar agar ukuran sama
            image = cv2.resize(image, (150,150))

            # menyimpan gambar ke variabel data
            data.append(image)

            # menyimpan label kelas
            labels.append(label)

        except:
            pass

# PREPROCESSING DATA

# normalisasi pixel dari 0-255 menjadi 0-1
data = np.array(data) / 255.0

# convert label menjadi array
labels = np.array(labels)

# mengubah label menjadi format kategorikal
labels = to_categorical(labels, 2)

# SPLIT DATASET
# 80% TRAINING DAN 20% TESTING


X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.20,
    random_state=42
)


# MEMBUAT ARSITEKTUR CNN
# SESUAI JURNAL


model = Sequential()

# convolution layer pertama menggunakan 32 filter
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))

# pooling layer untuk mengurangi dimensi data
model.add(MaxPooling2D(pool_size=(2,2)))

# convolution layer kedua menggunakan 64 filter
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# convolution layer ketiga menggunakan 96 filter
model.add(Conv2D(96, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# convolution layer keempat menggunakan 96 filter
model.add(Conv2D(96, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# flatten digunakan untuk mengubah feature map menjadi 1 dimensi
model.add(Flatten())

# fully connected layer
model.add(Dense(128, activation='relu'))

# output layer 2 kelas: mask dan no mask
model.add(Dense(2, activation='softmax'))

# menampilkan struktur model
model.summary()


# COMPILE MODEL


# konfigurasi optimizer, loss function, dan accuracy
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# TRAINING MODEL
# =========================================

# proses training CNN selama 50 epoch
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# =========================================
# EVALUASI MODEL
# =========================================

# menguji akurasi model
loss, accuracy = model.evaluate(X_test, y_test)

print("\n=== HASIL EVALUASI ===")
print("Loss     :", loss)
print("Accuracy :", accuracy)

# =========================================
# SAVE MODEL
# =========================================

# membuat folder models jika belum ada
if not os.path.exists("models"):
    os.makedirs("models")

# menyimpan model hasil training
model.save("models/mask_detector.h5")

print("\nModel berhasil disimpan!")

# =========================================
# VISUALISASI ACCURACY
# =========================================

# grafik akurasi training dan validation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.savefig("accuracy.png")
plt.show()

# =========================================
# VISUALISASI LOSS
# =========================================

# grafik loss training dan validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train", "Validation"])

plt.savefig("loss.png")
plt.show()