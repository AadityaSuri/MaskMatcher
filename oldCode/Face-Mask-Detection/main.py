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
from PIL import Image
import cv2
import asyncio


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
    
    
    cv2.imshow('Video', img)
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

cap = cv2.VideoCapture('http://10.0.0.29:8081/')



async def machineLearning():
    i=1
    while True:
        ret, cvFrame = cap.read()

        crop_img = cvFrame[0:480, 80:560]

        if (i%2 == 0):
            cv2.imwrite("webcamImage.png", cvFrame)
            print("Saved")
            
            prediction = isWearingMask(crop_img)
            print('Image predicted as', prediction)
            if (type(prediction) != type(-1)):
                print(prediction.detach().numpy()[0][0])
        
        i+=1
        await asyncio.sleep(0.1)

async def getFrames():
    while True:
        ret, cvFrame = cap.read()
        cv2.imshow('RAW Video', cvFrame)
        if cv2.waitKey(1) == 27:
            exit(0)
        await asyncio.sleep(0.01)


async def main():
    while True:
        f1 = loop.create_task(getFrames())
        f2 = loop.create_task(machineLearning())
        await asyncio.wait([f1, f2])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()