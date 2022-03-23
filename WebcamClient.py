import socket
from PIL import Image
import io
import time

serverHost = "127.0.0.1"  # The server's hostname or IP address
serverPort = 65432  # The port used by the server


with socket.socket() as s:
    print("Attempting to connect to server")
    s.connect((serverHost, serverPort))
    print("Connected to server")

    while True:
        with open('webcamImage.png', 'rb') as f:
            s.send(f.read())    
        print("sending")
        time.sleep(5)
