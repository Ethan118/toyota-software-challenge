import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from ultralytics import YOLO

cap = cv2.VideoCapture(1)

model = YOLO('C:/Users/ethan/Documents/Personal Projects/toyota-software-challenge/runs/segment/train28/weights/best.pt')

while True:
    ret, frame = cap.read()

    if not ret:
        print('Error: failed to capture image')
        break

    cv2.imshow('Raw', frame)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)

    # Map grayscale
    color_map = cv2.applyColorMap(gray, cv2.COLORMAP_PARULA)
    cv2.imshow('Color Map', color_map)

    # Detect objects
    results = model.predict(source=color_map, imgsz=(frame.shape[0], frame.shape[1]), conf=0.7, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow('Annotated', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

