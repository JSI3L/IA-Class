import os
import cv2
import numpy as np

# emociones_haarcascade.py
# Entrena y ejecuta un detector de emociones usando Haar Cascade para localizar caras
# y LBPH (OpenCV) como reconocedor basado en una "fuente de conocimiento" en ./datos/emociones
# Estructura esperada de ./datos/emociones:
#   ./datos/emociones/alegre/*.jpg
#   ./datos/emociones/triste/*.jpg
#   ./datos/emociones/enojado/*.png
# etc.


DATASET_DIR = os.path.join('.', 'datos', 'emociones')
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MODEL_PATH = os.path.join('.', 'modelo_emociones.yml')
FACE_SIZE = (200, 200)  # tamaño uniforme para el reconocedor
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

def list_image_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in ALLOWED_EXT]

def detect_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(faces) == 0:
        return None
    # tomar la cara más grande
    x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, FACE_SIZE)
    return face

def load_dataset(dataset_dir, face_cascade):
    faces = []
    labels = []
    label_names = []
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Directorio de datos no encontrado: {dataset_dir}")
    for i, emotion in enumerate(sorted(os.listdir(dataset_dir))):
        emotion_folder = os.path.join(dataset_dir, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        label_names.append(emotion)
        for path in list_image_files(emotion_folder):
            img = cv2.imread(path)
            if img is None:
                continue
            face = detect_face(img, face_cascade)
            if face is None:
                continue
            faces.append(face)
            labels.append(i)
    return faces, labels, label_names

def create_recognizer():
    # LBPH está en opencv contrib (cv2.face)
    if not hasattr(cv2, 'face'):
        raise RuntimeError("cv2.face no disponible. Instale opencv-contrib-python.")
    return cv2.face.LBPHFaceRecognizer_create()

def train_or_load_model():
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    recognizer = create_recognizer()
    label_names = []
    if os.path.exists(MODEL_PATH):
        try:
            recognizer.read(MODEL_PATH)
            # cargamos label_names desde un archivo paralelo si existe
            labels_file = MODEL_PATH + '.labels.txt'
            if os.path.exists(labels_file):
                with open(labels_file, 'r', encoding='utf-8') as f:
                    label_names = [l.strip() for l in f.readlines() if l.strip()]
            print("Modelo cargado desde", MODEL_PATH)
            return recognizer, label_names, face_cascade
        except Exception:
            print("No se pudo cargar el modelo. Se reentrenará.")
    # cargar dataset y entrenar
    faces, labels, label_names = load_dataset(DATASET_DIR, face_cascade)
    if len(faces) == 0:
        raise RuntimeError("No se encontraron caras entrenables en el dataset.")
    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)
    # guardar labels
    with open(MODEL_PATH + '.labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_names))
    print("Entrenamiento completado. Modelo guardado en", MODEL_PATH)
    return recognizer, label_names, face_cascade

def run_camera(recognizer, label_names, face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    print("Presione 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, FACE_SIZE)
            label_id, conf = recognizer.predict(roi)
            label_text = label_names[label_id] if 0 <= label_id < len(label_names) else f"id:{label_id}"
            # mostrar rectángulo y etiqueta
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            text = f"{label_text} ({conf:.0f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Emociones - Haarcascade + LBPH", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # flujo principal
    try:
        recognizer, label_names, face_cascade = train_or_load_model()
        run_camera(recognizer, label_names, face_cascade)
    except Exception as e:
        print("Error:", e)
        print("Asegúrese de tener ./datos/emociones con subcarpetas por emoción y opencv-contrib-python instalado.")