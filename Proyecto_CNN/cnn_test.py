import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image

# Cargar el modelo y las etiquetas
model = tf.keras.models.load_model('best_model.h5')
with open('class_labels.pkl', 'rb') as f:
    class_names = pickle.load(f)

print(f"Clases del modelo: {class_names}")

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Preprocesa una imagen para la predicción
    """
    img = Image.open(img_path)
    
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar
    img = img.resize(target_size)
    
    # Convertir a array y normalizar
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

def predict_animal(image_path):
    """
    Realiza la predicción en una imagen
    """
    # Preprocesar imagen
    img_array, original_img = preprocess_image(image_path)
    
    # Realizar predicción
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Obtener nombre de la clase
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, original_img

def display_prediction(image_path):
    """
    Muestra solo la imagen y el porcentaje de la clase predicha
    """
    predicted_class, confidence, img = predict_animal(image_path)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'{predicted_class} — {confidence:.2%}', fontsize=16)
    plt.show()
    
    return predicted_class, confidence

# Ejemplo de uso
if __name__ == "__main__":
    image_path = "dog5.jpeg"
    if os.path.exists(image_path):
        predicted_class, confidence = display_prediction(image_path)
    else:
        print(f"Imagen no encontrada: {image_path}")
        print("\nEjemplo de cómo usar:")
        print("1. Coloca tu imagen en la misma carpeta")
        print("2. Cambia 'test_image.jpg' por el nombre de tu imagen")
        print("3. Ejecuta: python predict.py")
