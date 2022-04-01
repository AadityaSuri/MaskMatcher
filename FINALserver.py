import threading
import time
import asyncio
import websockets

def FINALserver_thread():
    async def hello(websocket):
        print("WAITTTTTTTING")
        name = await websocket.recv()
        print(f"<<< {name}")

        greeting = f"Hello {name}!"

        await websocket.send(greeting)
        print(f">>> {greeting}")

        import globalVars
        while True:
            image = await websocket.recv()
            # print(image)
            print("Got image")

            with open('static/received_file123.png', 'wb') as f:
                f.write(image)
                
            prediction = await websocket.recv()
            print(prediction)
            globalVars.prediction = float(prediction)
            print("start thinks that " + prediction)
            
            await websocket.send(str(globalVars.doorLocked))


    async def main():
        async with websockets.serve(hello, "10.93.48.157", 443):
            await asyncio.Future() 


    def FINALserver():
        asyncio.run(main())

    thread1 = threading.Thread(target=FINALserver)
    thread1.start()