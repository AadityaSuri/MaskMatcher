import asyncio
from concurrent.futures import thread
import websockets
import cv2
import numpy as np
from PIL import Image
import time
import io
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import RPi.GPIO as GPIO
import pigpio


# BEGIN LED AND MOTOR VARIABLES AND CONFIG
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

REDLED = 22
GREENLED = 27
SERVO = 17


GPIO.setup(REDLED, GPIO.OUT)
GPIO.setup(GREENLED, GPIO.OUT)

val = -1

pwm = pigpio.pi() 
pwm.set_mode(SERVO, pigpio.OUTPUT)
 
pwm.set_PWM_frequency( SERVO, 50 )
pwm.set_servo_pulsewidth( SERVO, 500 ) ;


GPIO.output(REDLED,GPIO.LOW)
GPIO.output(GREENLED,GPIO.LOW)
# END LED AND MOTOR VARIABLES AND CONFIG

cap = cv2.VideoCapture('http://127.0.0.1:8081/')


# BEGIN NEURAL NETWORK FUNCTIONS
classes = ("with_mask", "without_mask")

# This is the class for the neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )


    def forward(self, x):
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        logits = self.linear_relu_stack(x)
        return logits


model = CNN()
model.load_state_dict(torch.load("NEWmodel.pth"))

model.eval()


# This function detects if the person in the image is wearing a mask or not
# It returns -1 if there is no person in the image
# Otherwise it returns the tensor values for the mask detection
def isWearingMask(cvImage):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    classes = ("with_mask", "without_mask")

    img = cvImage
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # This detects if there are eyes in the image using haardcascades
    eye_data = cv2.CascadeClassifier("haarcascade_eye.xml")
    found = eye_data.detectMultiScale(img_gray, 1.1, 4)

    amount_found = len(found)

    # This draws rectangles around the eyes found in the image
    for (x, y, w, h) in found:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite("eyes.png", img)
    
    
    cv2.imshow('NEW Video', img)
    if cv2.waitKey(1) == 27:
        exit(0)
    
    # If there are no eyes return 0
    # Otherwise, run the neural network on the image and return the result
    if amount_found != 0:

        img = Image.fromarray(cvImage)
        x = transform(img)
        x = x.unsqueeze(0)

        output = model(x)
        pred = F.softmax(output, dim=1)
        return pred
    else:
        return -1

# END NEURAL NETWORK FUNCTIONS



# BEGIN WEBSOCKET FUNCTIONS

# This is the main function that connects to the server and controls the servos and LEDs 
# based on the results of the machine learning function
async def connectToServer():
    uri = "ws://10.93.48.157:443"
    async with websockets.connect(uri) as websocket:
        # await asyncio.sleep(1)
        name = input("What's your name? ")
        

        await websocket.send(name)
        print(f">>> {name}")

        greeting = await websocket.recv()
        print(f"<<< {greeting}")

        i = 1
        while True:
            # Every second, the function sends images to the server
            if (i%10 == 0):
                # This reads the frame from the webcam and sends it via websockets to the server
                ret, cvFrame = cap.read()
                img = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
                pillowImage = Image.fromarray(img)

                img_byte_arr = io.BytesIO()
                pillowImage.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                await websocket.send(img_byte_arr)

                # This crops the image from the webcam and runs the machine learning model 
                # on the cropped image. It saves the result in the variable prediction
                crop_img = cvFrame[0:480, 80:560]
                prediction = isWearingMask(crop_img)
                print('Image predicted as', prediction)

                # This sends the machine learning prediction to the server
                if (type(prediction) != type(-1)):
                    await websocket.send(str(prediction.detach().numpy()[0][0]))
                else:
                    await websocket.send(str(-1))
                
                # This waits for the command to lock the door or unlock the door from the server 
                doorCommand = await websocket.recv()
                print("doorLocked is" + doorCommand)
                print(bool(doorCommand))

                # Based on the command to the door, it turns the LEDs on or off and turns the lock
                if doorCommand == "True":
                    print('door is locked')
                    # This is the middle position of the motor
                    pwm.set_servo_pulsewidth( SERVO, 1500 ) ;
                    GPIO.output(REDLED,GPIO.HIGH)
                    GPIO.output(GREENLED,GPIO.LOW)
                else:
                    print('door is NOT locked')
                    # This is the smallest position of the motor
                    pwm.set_servo_pulsewidth( SERVO, 500 ) ;
                    GPIO.output(REDLED,GPIO.LOW)
                    GPIO.output(GREENLED,GPIO.HIGH)

                print("Sent")
            
            await asyncio.sleep(0.1)
            i+=1
            # print(i)


# This function constantly get the frames from the livestream in order to avoid livestream delay
async def getFrames():
    while True:
        ret, cvFrame = cap.read()
        cv2.imshow('Video', cvFrame)
        if cv2.waitKey(1) == ord('␛'):
            exit(0)
        await asyncio.sleep(0.01)

# This is the wrapper for the two async functions: connectToServer and getFrames
async def connectToServerWrapper(loop):
    while True:
        f1 = loop.create_task(connectToServer())
        f2 = loop.create_task(getFrames())
        await asyncio.wait([f1, f2])

# This continuously runs the connectToServerWrapper function asyncronously 
def sendImages():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connectToServerWrapper(loop))
    loop.close()

# This begins the main thread used to do everything
if __name__ == "__main__":
    thread1 = Thread(target=sendImages)
    thread1.start()
    thread1.join()


# END WEBSOCKET FUNCTIONS