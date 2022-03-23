import socket
from PIL import Image
import time

serverHost = "127.0.0.1"  # Standard loopback interface address (localhost)
serverPort = 65432  # Port to listen on (non-privileged ports are > 1023)


with socket.socket() as s:
    s.bind((serverHost, serverPort))
    s.listen()

    print ("Waiting for the connection with client")
    conn, addr = s.accept()
    print("Connected to client")

    print(f"Connected by {addr}")
    with conn:
        while True:
            data = conn.recv(139264)
            print("Got image")

            # data.show()
            
            if not data:
                break

            with open('received_file.png', 'wb') as f:
                f.write(data)
            f.close()
            time.sleep(5)