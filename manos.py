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
        cv2.circle(frame, (x, y), 5, (233, 23, 0), -1)  # Puntos
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Dibujar coordenadas del pulgar e índice
    cv2.putText(frame, f'({int(pulgar[0])}, {int(pulgar[1])})', (pulgar[0], pulgar[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f'({int(indice[0])}, {int(indice[1])})', (indice[0], indice[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 0, 0), 2, cv2.LINE_AA)

    cv2.line(frame, (int(pulgar[0]), int(pulgar[1])), (int(indice[0]), int(indice[1])), (244,34,12), 2)
    
    # Calcular distancias en píxeles
    distancia_pulgar_indice = np.linalg.norm(np.array(pulgar) - np.array(indice))

    cv2.putText(frame, f'({int(distancia_pulgar_indice)})', (pulgar[0]-40, pulgar[1] - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    centro_x = w // 2
    centro_y = h // 2

    # Calcular el tamaño del rectángulo (usando la distancia como referencia)
    tamaño = max(10, int(distancia_pulgar_indice))  # mínimo para que no sea cero
    
    # Calcular ángulo de rotación en función de la posición "y" del dedo índice
    # Mapeamos la diferencia vertical a un ángulo en grados dentro de [-45, 45]
    max_angle = 180.0
    dy = indice[1] - centro_y
    angle = np.clip((dy / float(centro_y)) * max_angle, -max_angle, max_angle)

    # Crear rectángulo rotado centrado en (centro_x, centro_y)
    # boxPoints recibe ((cx,cy),(width,height), angle)
    box = cv2.boxPoints(((centro_x, centro_y), (tamaño, tamaño), angle))
    box = np.int32(box)

    # Dibujar rectángulo rotado
    cv2.polylines(frame, [box], isClosed=True, color=(34, 234, 65), thickness=3)

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