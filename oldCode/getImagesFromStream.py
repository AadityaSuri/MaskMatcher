import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps   
cap = cv2.VideoCapture('http://192.168.107.182:8081/')

while True:
    ret, cvFrame = cap.read()

    cv2.imshow('Video', cvFrame)
    if cv2.waitKey(1) == 27:
        exit(0)