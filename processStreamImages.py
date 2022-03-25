import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps   
import time
cap = cv2.VideoCapture('http://192.168.92.182:8081/')


i=0
while True:
    ret, cvFrame = cap.read()

    img = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
    pillowImage = Image.fromarray(img)

    # Process the image using PIL

    # pillowImage = PIL.ImageOps.invert(pillowImage)

    # pillowImage.show()
    # pillowSaved = pillowImage.save(f"images/webcamFrame{i}.png")
    pillowSaved = pillowImage.save(f"webcamImage.png")

    cvImage = cv2.cvtColor(np.asarray(pillowImage), cv2.COLOR_RGB2BGR) 
    cv2.imshow('Video', cvImage)
    if cv2.waitKey(1) == 27:
        exit(0)
    
    i+=1