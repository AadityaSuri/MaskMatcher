import asyncio
import websockets

async def hello():
    uri = "ws://10.93.48.157:8765"
    async with websockets.connect(uri) as websocket:
        name = input("What's your name? ")

        await websocket.send(name)
        print(f">>> {name}")

        greeting = await websocket.recv()
        print(f"<<< {greeting}")

if __name__ == "__main__":
    asyncio.run(hello())

# PI: 192.168.92.182

# Server:  10.93.49.254

# Maybe server: 10.93.48.157