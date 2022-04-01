import asyncio
import websockets
import cv2
import numpy as np
from PIL import Image
import time
from io import BytesIO
import io

cap = cv2.VideoCapture('http://192.168.8.165:8081/')

async def hello():
    uri = "ws://10.93.48.157:443"
    async with websockets.connect(uri) as websocket:
        # await asyncio.sleep(1)
        name = input("What's your name? ")
        

        await websocket.send(name)
        print(f">>> {name}")

        greeting = await websocket.recv()
        print(f"<<< {greeting}")

        while True:
            # with open('webcamimage.png', 'rb') as f:
            #     await websocket.send(f.read())
            #     print("Sent image")
            ret, cvFrame = cap.read()
            img = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
            pillowImage = Image.fromarray(img)

            img_byte_arr = io.BytesIO()
            pillowImage.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            await websocket.send(img_byte_arr)

            # await websocket.send(cvFrame)
            print("Sent")
            
            await asyncio.sleep(0.1)

async def getFrames():
    while True:
        ret, cvFrame = cap.read()
        cv2.imshow('Video', cvFrame)
        if cv2.waitKey(1) == 27:
            exit(0)
        await asyncio.sleep(0.01)


async def main():
    while True:
        f1 = loop.create_task(hello())
        f2 = loop.create_task(getFrames())
        await asyncio.wait([f1, f2])
        # await asyncio.wait([f1])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

# PI: 192.168.92.182

# Server:  10.93.49.254

# Maybe server: 10.93.48.157