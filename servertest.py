import socket
from PIL import Image

serverHost = "127.0.0.1"  # Standard loopback interface address (localhost)
serverPort = 65432  # Port to listen on (non-privileged ports are > 1023)


with open('received_file.png', 'wb') as f:
    print("file opened")
    with socket.socket() as s:
        s.bind((serverHost, serverPort))
        s.listen()

        print ("Waiting for the connection with client")
        conn, addr = s.accept()
        print("Connected to client")

        print(f"Connected by {addr}")
        with conn:
            while True:
                data = conn.recv(114688)
                print("HELLOOOOO")

                # data.show()

                if not data:
                    break
                f.write(data)
                print(data)
                conn.sendall(b"\nServer says hello too")

f.close()