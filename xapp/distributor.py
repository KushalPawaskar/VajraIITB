import socket
import numpy as np
import time


XAPP_IP_ADDR = "127.0.0.1"
XAPP_PORT = 5001

while True:
    xapp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    xapp_sock.connect((XAPP_IP_ADDR, XAPP_PORT))

    vect = np.random.rand(36).tolist()
    s = ""
    for i in vect:
        if i >= 0.5:
            s += "1 "
        else:
            s += "0 "
    s = s[:-1]

    xapp_sock.send(s.encode())

    xapp_sock.close()

    time.sleep(15*60)
