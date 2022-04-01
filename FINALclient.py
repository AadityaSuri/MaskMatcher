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


# LED AND MOTOR VARIABLES AND CONFIG
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


cap = cv2.VideoCapture('http://127.0.0.1:8081/')

# BEGIN NEURAL NETWORK FUNCTIONS

classes = ("with_mask", "without_mask")

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


def isWearingMask(cvImage):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    classes = ("with_mask", "without_mask")

    img = cvImage
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eye_data = cv2.CascadeClassifier("haarcascade_eye.xml")
    found = eye_data.detectMultiScale(img_gray, 1.1, 4)

    amount_found = len(found)

    for (x, y, w, h) in found:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite("eyes.png", img)
    
    
    cv2.imshow('NEW Video', img)
    if cv2.waitKey(1) == 27:
        exit(0)

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
            if (i%10 == 0):
                ret, cvFrame = cap.read()
                img = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
                pillowImage = Image.fromarray(img)

                img_byte_arr = io.BytesIO()
                pillowImage.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                await websocket.send(img_byte_arr)

                crop_img = cvFrame[0:480, 80:560]
                prediction = isWearingMask(crop_img)
                print('Image predicted as', prediction)

                if (type(prediction) != type(-1)):
                    await websocket.send(str(prediction.detach().numpy()[0][0]))
                else:
                    await websocket.send(str(-1))
                
                doorCommand = await websocket.recv()
                print("doorLocked is" + doorCommand)
                print(bool(doorCommand))
                if doorCommand == "True":
                    print('door is locked')
                    pwm.set_servo_pulsewidth( SERVO, 1500 ) ;
                    GPIO.output(REDLED,GPIO.HIGH)
                    GPIO.output(GREENLED,GPIO.LOW)
                else:
                    print('door is NOT locked')
                    pwm.set_servo_pulsewidth( SERVO, 500 ) ;
                    GPIO.output(REDLED,GPIO.LOW)
                    GPIO.output(GREENLED,GPIO.HIGH)

                print("Sent")
            
            await asyncio.sleep(0.1)
            i+=1
            # print(i)


async def getFrames():
    while True:
        ret, cvFrame = cap.read()
        cv2.imshow('Video', cvFrame)
        if cv2.waitKey(1) == ord('␛'):
            exit(0)
        await asyncio.sleep(0.01)

async def connectToServerWrapper(loop):
    while True:
        f1 = loop.create_task(connectToServer())
        f2 = loop.create_task(getFrames())
        await asyncio.wait([f1, f2])

def sendImages():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connectToServerWrapper(loop))
    loop.close()

if __name__ == "__main__":
    thread1 = Thread(target=sendImages)
    # thread2 = Thread(target=getFramesWrapper,args=(cap,))
    thread1.start()
    # thread2.start()
    thread1.join()
    # thread2.join()