import socket
from PIL import Image
import io

serverHost = "127.0.0.1"  # The server's hostname or IP address
serverPort = 65432  # The port used by the server

with socket.socket() as s:

    print("Attempting to connect to server")
    s.connect((serverHost, serverPort))
    print("Connected to server")

    # testImage = Image.open("testimage.jpg")
    # testImage.show()

    # img_byte_arr = io.BytesIO()
    # testImage.save(img_byte_arr, format='PNG')

    # print("Sending image")

    # byteImageSending = img_byte_arr.read(1024)


    with open('testimage.jpg', 'rb') as f:
        s.send(f.read())    
    print("actually sending")
    # s.sendall(byteImageSending)
    data = s.recv(1024)

print(b"Received: " + data)