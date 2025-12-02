import cv2 as cv
import numpy as np
import os

print("[1] Loading dataset...")
dataSet = "C:\\Dev\\IA-Class\\jsiel"

faces = os.listdir(dataSet)
print(f"[1.1] Found folders (people): {faces}")

labels = []
facesData = []
label = 0

print("[2] Reading images and assigning labels...")
for face in faces:
    facePath = os.path.join(dataSet, face)
    print(f"  -> Processing folder: {facePath}")

    for faceName in os.listdir(facePath):
        imgPath = os.path.join(facePath, faceName)
        img = cv.imread(imgPath, 0)

        if img is None:
            print(f"    [WARNING] Could not read {imgPath}")
            continue

        labels.append(label)
        facesData.append(img)
    print(f"  -> {len(os.listdir(facePath))} images processed for label {label}")

    label += 1

print("[3] Total faces loaded:", len(facesData))
print("[3.1] Labels count:", len(labels))
print("[3.2] Label 0 count:", np.count_nonzero(np.array(labels) == 0))

print("[4] Training EigenFaceRecognizer...")
faceRecognizer = cv.face.EigenFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))

output_file = "Eigenface80.xml"
print(f"[5] Saving model to {output_file}...")
faceRecognizer.write(output_file)

print("[6] Done ✅ Model saved successfully.")