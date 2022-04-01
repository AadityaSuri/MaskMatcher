import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps   
import time
cap = cv2.VideoCapture('http://192.168.8.235:8081/')


i=0
while True:
    ret, cvFrame = cap.read()

    img = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
    pillowImage = Image.fromarray(img)

    # Process the image using PIL

    pillowImage = pillowImage.crop((80,0,560,480))

    # pillowImage.show()
    # pillowSaved = pillowImage.save(f"images/webcamFrame{i}.png")
    if (i%10 == 0):
        pillowSaved = pillowImage.save(f"webcamImage.png")
        print("Saved")

    cvImage = cv2.cvtColor(np.asarray(pillowImage), cv2.COLOR_RGB2BGR) 
    cv2.imshow('Video', cvImage)
    if cv2.waitKey(1) == ord('s'):
        exit(0)
    
    i+=1