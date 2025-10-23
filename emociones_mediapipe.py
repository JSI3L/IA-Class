import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def flatten_pairs(pairs):
    # extrae índices únicos de las tuplas de conexión de mediapipe
    s = set()
    for a, b in pairs:
        s.add(a)
        s.add(b)
    return sorted(list(s))

def landmarks_to_array(landmarks, image_w, image_h):
    pts = []
    for lm in landmarks:
        pts.append((lm.x * image_w, lm.y * image_h))
    return np.array(pts)

def center_of(pts):
    if len(pts) == 0:
        return np.array([0.0, 0.0])
    return np.mean(pts, axis=0)

def bbox_of(pts):
    if len(pts) == 0:
        return (0,0,0,0)
    min_x = np.min(pts[:,0])
    max_x = np.max(pts[:,0])
    min_y = np.min(pts[:,1])
    max_y = np.max(pts[:,1])
    return (min_x, min_y, max_x, max_y)

def classify_emotion(mouth_w, mouth_h, inter_eye, brows_dist, brow_eye_gap_l, brow_eye_gap_r):
    # Heurísticas simples y ajustables
    # Todos los valores vienen normalizados por el ancho/alto de la cara
    if mouth_w > 0.45:
        return "Happy"
    if mouth_h > 0.20 and inter_eye > 0.30:
        return "Surprised"
    if brows_dist < 0.18 and (brow_eye_gap_l < 0.04 or brow_eye_gap_r < 0.04):
        return "Angry"
    if mouth_w < 0.30 and mouth_h < 0.06 and (brow_eye_gap_l > 0.06 and brow_eye_gap_r > 0.06):
        return "Sad"
    return "Neutral"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    # índices por región usando las constantes de face_mesh
    lips_idx = flatten_pairs(mp_face.FACEMESH_LIPS)
    left_eye_idx = flatten_pairs(mp_face.FACEMESH_LEFT_EYE)
    right_eye_idx = flatten_pairs(mp_face.FACEMESH_RIGHT_EYE)
    left_brow_idx = flatten_pairs(mp_face.FACEMESH_LEFT_EYEBROW)
    right_brow_idx = flatten_pairs(mp_face.FACEMESH_RIGHT_EYEBROW)
    face_oval_idx = flatten_pairs(mp_face.FACEMESH_FACE_OVAL)

    face_mesh = mp_face.FaceMesh(static_image_mode=False,
                                 max_num_faces=1,
                                 refine_landmarks=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        emotion = "No face"
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            pts = landmarks_to_array(lm, w, h)

            # obtener regiones
            lips = pts[lips_idx]
            left_eye = pts[left_eye_idx]
            right_eye = pts[right_eye_idx]
            left_brow = pts[left_brow_idx]
            right_brow = pts[right_brow_idx]
            face_oval = pts[face_oval_idx]

            # bbox / tamaños de referencia
            face_min_x, face_min_y, face_max_x, face_max_y = bbox_of(face_oval)
            face_width = max(1.0, face_max_x - face_min_x)
            face_height = max(1.0, face_max_y - face_min_y)

            # métricas de la boca
            mouth_min_x, mouth_min_y, mouth_max_x, mouth_max_y = bbox_of(lips)
            mouth_w = (mouth_max_x - mouth_min_x) / face_width
            mouth_h = (mouth_max_y - mouth_min_y) / face_height

            # ojos: centros e interocular
            left_eye_c = center_of(left_eye)
            right_eye_c = center_of(right_eye)
            inter_eye = np.linalg.norm(left_eye_c - right_eye_c) / face_width

            # cejas: centros y distancia entre cejas (horizontal)
            left_brow_c = center_of(left_brow)
            right_brow_c = center_of(right_brow)
            brows_dist = abs(left_brow_c[0] - right_brow_c[0]) / face_width

            # separación vertical entre ceja y ojo (si ceja baja -> gap pequeño)
            brow_eye_gap_l = (left_eye_c[1] - left_brow_c[1]) / face_height
            brow_eye_gap_r = (right_eye_c[1] - right_brow_c[1]) / face_height

            emotion = classify_emotion(mouth_w, mouth_h, inter_eye, brows_dist, brow_eye_gap_l, brow_eye_gap_r)

            # Dibujar información en la imagen
            cv2.rectangle(frame, (int(face_min_x), int(face_min_y)), (int(face_max_x), int(face_max_y)), (0,255,0), 1)
            cv2.putText(frame, f"Emotion: {emotion}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            # métricas secundarias
            cv2.putText(frame, f"mouth_w:{mouth_w:.2f} mouth_h:{mouth_h:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(frame, f"inter_eye:{inter_eye:.2f} brows_dist:{brows_dist:.2f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # puntos guía
            for p in (lips, left_eye, right_eye, left_brow, right_brow):
                for (x,y) in p.astype(int):
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 1)

        cv2.imshow("Emociones - Mediapipe", frame)
        key = cv2.waitKey(1) & 0xFF
        # permitir salir con ESC o q/Q
        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()