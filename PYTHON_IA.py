import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def carregar_modelo():
    model = keras.applications.MobileNetV2(weights='imagenet')
    return model

def preprocessar_imagem(img):
    img = cv2.resize(img, (224, 224))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = keras.applications.mobilenet_v2.preprocess_input(img) 
    img = np.expand_dims(img, axis=0)  
    return img

def fazer_previsao(model, img):
    previsao = model.predict(img)
    return previsao

def mostrar_resultados(previsao):
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(previsao, top=3)[0]
    result_text = "\n".join([f"{pred[1]}: {pred[2]*100:.2f}%" for pred in decoded_predictions])
    return result_text

def carregar_imagem():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    img = cv2.imread(filepath)
    if img is None:
        result_label.config(text="Erro: não foi possível carregar a imagem. Verifique o caminho do arquivo.")
        return
    
    imagem_preprocessada = preprocessar_imagem(img)
    previsao = fazer_previsao(modelo, imagem_preprocessada)
    resultados = mostrar_resultados(previsao)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    
    result_label.config(text=resultados)

root = tk.Tk()
root.title("Classificador de Imagens com MobileNetV2")

carregar_btn = tk.Button(root, text="Carregar Imagem", command=carregar_imagem)
carregar_btn.pack()

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="", justify=tk.LEFT, padx=10, pady=10)
result_label.pack()

modelo = carregar_modelo()

root.mainloop()
