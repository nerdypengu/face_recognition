import cv2
import numpy as np
import os

def checkDataset(directory="dataset/"):
    if os.path.exists(directory) and len(os.listdir(directory)) != 0:
        return True
    return False

def organizeDataset(path="dataset/"):
    imagePath = [os.path.join(path, p) for p in os.listdir(path)]
    faces = []
    ids = np.array([], dtype="int")
    
    for i in imagePath:
        img = cv2.imread(i, cv2.COLOR_BGR2GRAY)
        id = int(i.split(".")[0].split("-")[1])
        face = faceCascade.detectMultiScale(img)
        for (x, y, w, h) in face:
            faces.append(img[y:y+h, x:x+w])
            ids = np.append(ids, id)
    return faces, ids

if not checkDataset():
    print("Dataset not found")
else:
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    # Train faces
    print("Training faces...")
    faces, ids = organizeDataset()
    recognizer.train(faces, ids)
    print("Training finished!")
    
    # Save model
    recognizer.write("face-model.yml")
    print("Model saved as 'face-model.yml'")
