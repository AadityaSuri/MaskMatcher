import asyncio
import websockets
import cv2
import numpy as np
from PIL import Image
import time

async def hello():
    uri = "ws://10.93.48.157:443"
    async with websockets.connect(uri) as websocket:
        name = input("What's your name? ")

        await websocket.send(name)
        print(f">>> {name}")

        greeting = await websocket.recv()
        print(f"<<< {greeting}")

        while True:
            with open('webcamimage.png', 'rb') as f:
                await websocket.send(f.read())
                print("Sent image")
            
            time.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(hello())

# PI: 192.168.92.182

# Server:  10.93.49.254

# Maybe server: 10.93.48.157