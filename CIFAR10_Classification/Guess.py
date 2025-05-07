import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model

class_labels = ['Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse']

model = load_model("cnn_model_cifar6.h5")

# 32x32 ve normalize
def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32))
    img = np.array(img) / 255.0
    if img.shape != (32, 32, 3):
        raise ValueError("Resim 3 kanallı değil!")
    return np.expand_dims(img, axis=0)

# Tahmin fonksiyonu
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            # Görseli göster ve sonucu yaz
            img = Image.open(file_path).resize((128, 128))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            result_label.config(text=f"Tahmin: {predicted_class}")

        except Exception as e:
            result_label.config(text=f"Hata: {str(e)}")

# Ana pencere
root = tk.Tk()
root.title("Görsel Tanıma")
root.geometry("400x500")
root.configure(bg="#f0f0f0")

# Başlık
title_label = tk.Label(root, text="Görsel Tanıma", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333")
title_label.pack(pady=20)

# Resim seç butonu
btn = tk.Button(
    root,
    text="📁 Resim Seç ve Tahmin Et",
    command=predict_image,
    font=("Helvetica", 12),
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5,
    relief="groove"
)
btn.pack(pady=10)

# Görsel görüntüleme alanı
image_label = tk.Label(root, bg="#f0f0f0")
image_label.pack(pady=10)

# Sonuç metni
result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f0f0f0", fg="#555")
result_label.pack(pady=20)

root.mainloop()