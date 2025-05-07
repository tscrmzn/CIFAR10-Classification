import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model, Sequential

# CIFAR-10 veri setini yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Kullanılacak sınıflar: bird (2), cat (3), deer (4), dog (5), frog (6), horse (7)
allowed_classes = [2, 3, 4, 5, 6, 7]

# Bu sınıflara ait verileri filtreleme
train_mask = np.isin(y_train, allowed_classes).flatten()
test_mask = np.isin(y_test, allowed_classes).flatten()

x_train = x_train[train_mask]
y_train = y_train[train_mask]
x_test = x_test[test_mask]
y_test = y_test[test_mask]

# Etiketleri yeniden numaralandır (0-5 arası)
label_map = {label: idx for idx, label in enumerate(allowed_classes)}
y_train = np.array([label_map[y[0]] for y in y_train])
y_test = np.array([label_map[y[0]] for y in y_test])

# One-hot encode
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)

# Normalize (0-255 → 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model_path = "cnn_model_cifar6.h5"

if os.path.exists(model_path):
    print(" Kayıtlı model bulundu, yükleniyor...")
    cnn = load_model(model_path)
else:
    print(" Yeni model oluşturuluyor...")

    # CNN Modeli
    cnn = Sequential()
    cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
    cnn.add(MaxPooling2D(pool_size=2))

    cnn.add(Conv2D(128, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=2))

    cnn.add(Conv2D(256, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=2))

    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(256, activation='relu'))
    cnn.add(Dropout(0.3))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dense(6, activation='softmax'))  # 6 sınıf

    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Eğitim
    cnn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=64)

    # Modeli kaydetme
    cnn.save(model_path)
    print(f"✅ Model kaydedildi: {model_path}")
