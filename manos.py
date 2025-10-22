import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Función para determinar la letra según la posición de los dedos
def reconocer_letra(hand_landmarks, frame):
    h, w, _ = frame.shape  # Tamaño de la imagen
    
    # Obtener coordenadas de los puntos clave en píxeles
    dedos = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in range(21)]
    
    # Obtener posiciones clave (puntas de los dedos)
    pulgar, indice  = dedos[4], dedos[8] 

    # Mostrar los números de los landmarks en la imagen
    for i, (x, y) in enumerate(dedos):
        cv2.circle(frame, (x, y), 5, (233, 23, 0), -1)  # Puntos verdes
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Dibujar coordenadas del pulgar
    cv2.putText(frame, f'({int(pulgar[0])}, {int(pulgar[1])})', (pulgar[0], pulgar[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f'({int(indice[0])}, {int(indice[1])})', (indice[0], indice[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 0, 0), 2, cv2.LINE_AA)

    cv2.line(frame, (int(pulgar[0]), int(pulgar[1])), (int(indice[0]), int(indice[1])), (244,34,12), 2)
    
    # Calcular distancias en píxeles
    distancia_pulgar_indice = np.linalg.norm(np.array(pulgar) - np.array(indice))

    cv2.putText(frame, f'({distancia_pulgar_indice})', (pulgar[0]-40, pulgar[1] - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    centro_x = w // 2
    centro_y = h // 2
    
    # Calcular el tamaño del rectángulo (usando la distancia como referencia)
    tamaño = int(distancia_pulgar_indice)
    
    # Calcular las coordenadas del rectángulo centrado
    x1 = centro_x - tamaño // 2
    y1 = centro_y - tamaño // 2
    x2 = centro_x + tamaño // 2
    y2 = centro_y + tamaño // 2
    
    # Dibujar rectángulo en el centro del frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 234, 65), 3)
    

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(frame_rgb)

    # Dibujar puntos de la mano y reconocer letras
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Identificar la letra
            letra_detectada = reconocer_letra(hand_landmarks, frame)

            
    # Mostrar el video
    cv2.imshow("Rect", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()