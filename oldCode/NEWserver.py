import asyncio
import websockets

async def hello(websocket):
    name = await websocket.recv()
    print(f"<<< {name}")

    greeting = f"Hello {name}!"

    await websocket.send(greeting)
    print(f">>> {greeting}")

    while True:
        image = await websocket.recv()
        # print(image)
        print("Got image")

        with open('received_file123.png', 'wb') as f:
            f.write(image)

async def main():
    async with websockets.serve(hello, "10.93.48.157", 443):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
