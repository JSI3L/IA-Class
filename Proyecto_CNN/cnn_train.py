import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Verificar y crear estructura de directorios
def setup_directories():
    base_dirs = ['data/train', 'data/validation']
    classes = ['catarina', 'hormiga', 'perro', 'gato', 'tortuga']
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Creando directorio: {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
        
        for class_name in classes:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Creando directorio de clase: {class_dir}")
                os.makedirs(class_dir, exist_ok=True)
    
    print("\nEstructura de directorios creada.")
    print("Por favor, coloca tus imágenes en las carpetas correspondientes:")
    print("  - data/train/[clase]/")
    print("  - data/validation/[clase]/")
    print("\nClases: catarina, hormiga, perro, gato, tortuga")

# Configuración
IMG_SIZE = (128, 128)
BATCH_SIZE = 16  # Reducido para datasets pequeños
EPOCHS = 15  # Reducido temporalmente
NUM_CLASSES = 5

# Rutas del dataset
train_dir = 'data/train'
validation_dir = 'data/validation'

# Verificar si existen los directorios
if not os.path.exists(train_dir):
    print(f"ERROR: No se encuentra el directorio: {train_dir}")
    print("\nCreando estructura de directorios...")
    setup_directories()
    print("\nPor favor, coloca tus imágenes en las carpetas y vuelve a ejecutar.")
    sys.exit(1)

# Verificar imágenes en cada directorio
def count_images_in_directory(directory_path):
    """Cuenta imágenes en un directorio y sus subdirectorios"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    count = 0
    class_counts = {}
    
    if not os.path.exists(directory_path):
        return 0, {}
    
    for class_name in os.listdir(directory_path):
        class_path = os.path.join(directory_path, class_name)
        if os.path.isdir(class_path):
            class_images = 0
            for file in os.listdir(class_path):
                if file.lower().endswith(image_extensions):
                    class_images += 1
                    count += 1
            class_counts[class_name] = class_images
    
    return count, class_counts

print("=== VERIFICACIÓN DE IMÁGENES ===")
train_count, train_class_counts = count_images_in_directory(train_dir)
val_count, val_class_counts = count_images_in_directory(validation_dir)

print(f"\nImágenes en train: {train_count}")
for class_name, count in train_class_counts.items():
    print(f"  {class_name}: {count} imágenes")

print(f"\nImágenes en validation: {val_count}")
for class_name, count in val_class_counts.items():
    print(f"  {class_name}: {count} imágenes")

# Verificar si hay suficientes imágenes
MIN_IMAGES_PER_CLASS = 5  # Mínimo absoluto
MIN_TOTAL_IMAGES = 20

if train_count == 0:
    print("\n❌ ERROR: No hay imágenes en el directorio de entrenamiento.")
    print("Por favor, agrega imágenes en formato JPG, PNG, etc.")
    print("Ejemplo de estructura esperada:")
    print("  data/train/catarina/imagen1.jpg")
    print("  data/train/catarina/imagen2.jpg")
    print("  ... etc.")
    sys.exit(1)

if val_count == 0:
    print("\n⚠️ ADVERTENCIA: No hay imágenes en el directorio de validación.")
    print("Usando todas las imágenes para entrenamiento (sin validación separada).")
    validation_dir = train_dir  # Usar mismo directorio para validación

# Ajustar batch_size si hay pocas imágenes
if train_count < BATCH_SIZE * 2:
    BATCH_SIZE = max(4, train_count // 2)  # Batch_size mínimo de 4
    print(f"\n⚠️ Dataset pequeño. Ajustando batch_size a: {BATCH_SIZE}")

# Verificar clases
classes_in_train = set(train_class_counts.keys())
if len(classes_in_train) < 2:
    print(f"\n❌ ERROR: Se necesitan al menos 2 clases diferentes.")
    print(f"Clases encontradas: {list(classes_in_train)}")
    sys.exit(1)

print(f"\n✅ Dataset verificado: {train_count} imágenes de entrenamiento, {val_count} imágenes de validación")

# Data augmentation para entrenamiento
print("\nConfigurando data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 if val_count == 0 else 0.0  # Split si no hay validation separado
)

# Solo reescalar para validación
validation_datagen = ImageDataGenerator(rescale=1./255)

print("Creando generadores de datos...")
try:
    # Si no hay directorio de validación separado, usar split del train
    if val_count == 0:
        print("Usando 80% para entrenamiento, 20% para validación...")
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
    else:
        # Usar directorios separados
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
    
    print("✅ Generadores creados exitosamente")
    
except Exception as e:
    print(f"❌ Error al crear generadores: {e}")
    print("\nPosibles soluciones:")
    print("1. Verifica que cada carpeta de clase tiene al menos una imagen")
    print("2. Asegúrate de que las imágenes tienen formatos válidos (.jpg, .png, etc.)")
    print("3. Intenta con menos clases si tienes pocas imágenes")
    sys.exit(1)

# Verificar las clases encontradas
class_names = list(train_generator.class_indices.keys())
print(f"\n✅ Clases encontradas: {class_names}")
print(f"📊 Total de clases: {len(class_names)}")
print(f"📊 Número de imágenes de entrenamiento: {train_generator.samples}")
print(f"📊 Número de imágenes de validación: {validation_generator.samples}")

# Ajustar steps por época
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

print(f"\n⚙️  Configuración de entrenamiento:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Steps por época: {steps_per_epoch}")
print(f"   Validation steps: {validation_steps}")

# Construcción del modelo CNN más simple para dataset pequeño
print("\nConstruyendo modelo CNN...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dropout(0.3),  # Reducido para dataset pequeño
    layers.Dense(128, activation='relu'),  # Reducido
    layers.Dense(len(class_names), activation='softmax')
])

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Tasa de aprendizaje más baja
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo construido y compilado")
model.summary()

# Callbacks para mejorar el entrenamiento
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Entrenamiento
print(f"\n🚀 Iniciando entrenamiento por {EPOCHS} épocas...")
print("   Esto puede tomar unos minutos...")

try:
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Entrenamiento completado exitosamente!")
    
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {e}")
    print("\n💡 Posibles soluciones:")
    print("   1. Reduce aún más el batch_size (ej: 8 o 4)")
    print("   2. Reduce el tamaño de las imágenes (ej: 100x100)")
    print("   3. Agrega más imágenes a tu dataset")
    print("   4. Verifica que todas las imágenes se puedan leer")
    
    # Intentar con batch_size más pequeño
    print("\n🔄 Intentando con batch_size=8...")
    BATCH_SIZE = 8
    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)
    
    train_generator.batch_size = BATCH_SIZE
    validation_generator.batch_size = BATCH_SIZE
    
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,  # Menos épocas
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        print("✅ Entrenamiento completado con batch_size=8!")
    except Exception as e2:
        print(f"❌ Error persistente: {e2}")
        print("\n🎯 Solución definitiva: Agrega más imágenes a tu dataset.")
        print("   Necesitas al menos 10-20 imágenes por clase para empezar.")
        sys.exit(1)

# Guardar el modelo final
model.save('animal_classifier_model.h5')
print("💾 Modelo guardado como 'animal_classifier_model.h5'")

# Guardar las etiquetas de clase
import pickle
with open('class_labels.pkl', 'wb') as f:
    pickle.dump(class_names, f)
print("🏷️  Etiquetas guardadas como 'class_labels.pkl'")

# Gráficas de precisión y pérdida
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión entrenamiento', marker='o')
plt.plot(history.history['val_accuracy'], label='Precisión validación', marker='s')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida entrenamiento', marker='o')
plt.plot(history.history['val_loss'], label='Pérdida validación', marker='s')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
print("📊 Gráficas guardadas como 'training_history.png'")
plt.show()

# Evaluación final
print("\n📈 Evaluando modelo final...")
test_loss, test_acc = model.evaluate(validation_generator, verbose=0)
print(f"🎯 Precisión en validación: {test_acc:.2%}")

# Mostrar resumen
print("\n" + "="*50)
print("🎉 RESUMEN DEL ENTRENAMIENTO")
print("="*50)
print(f"📁 Clases: {', '.join(class_names)}")
print(f"📊 Imágenes de entrenamiento: {train_count}")
print(f"📊 Imágenes de validación: {val_count if val_count > 0 else '20% del train'}")
print(f"🎯 Precisión final: {test_acc:.2%}")
print(f"💾 Modelo guardado: animal_classifier_model.h5")
print(f"🏷️  Etiquetas guardadas: class_labels.pkl")
print("="*50)
print("\n✅ ¡Listo! Ahora puedes usar el modelo para predecir.")
print("   Ejecuta: python predict.py")