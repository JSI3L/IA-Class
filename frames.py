import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

save_dir = "jsiel"
os.makedirs(save_dir, exist_ok=True)
face_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = frame[y : y + h, x : x + w]
        face_img = cv2.resize(face_img, (200, 200))
        face_path = os.path.join(save_dir, f"face_{face_id}.jpg")

        cv2.imwrite(face_path, face_img)
        face_id += 1

    cv2.imshow("detected faces", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
