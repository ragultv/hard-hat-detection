from ultralytics import YOLO
import cv2
# Create a new YOLO model from scratch
model = YOLO("best.pt")
image=cv2.imread('hardhat5.jpg')
results = model(image)
print(results)
detected=results[0].plot()
cv2.imshow('detected',detected)
cv2.waitKey(0)
