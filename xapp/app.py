import socket


def get_BS_by_region(region):
    return []


def get_PWS_IWF_servers(region):
    return []


def send_e2(region, base_stations, status, commands):
    if status == 1:
        print(region, "base stations:", "Heavy Rain", commands)
    else:
        print(region, "base stations:", "Normal Rain", commands)


def send_e2_pws(region, pws_servers, status, commands):
    print(region, "PWS-IWF servers:", "Heavy Rain", commands)


cmds = {
    "HIGH_PRIORITY_RESOURCE_ALLOCATION": "HIGH_PRIORITY",
    "NORMAL_RESOURCE_ALLOCATION": "NORMAL_PRIORITY",
    "BROADCAST_MESSAGE": "BROADCAST_ALERT",
}

regions = ['Andheri', 'B ward', 'Bandra', 'Byculla', 'C ward', 'Chembur', 
           'Chincholi', 'Colaba', 'D Ward', 'Dahisar', 'Dindoshi', 'F North',
           'F South', 'G South', 'Gowanpada', 'H West ward', 'K East ward',
           'K West ward', 'Kandivali', 'Kurla', 'L ward', 'M West ward', 'MCGM 1',
           'Malvani', 'Marol', 'Memonwada', 'Mulund', 'N ward', 'Nariman Fire',
           'Rawali camp', 'S ward', 'SWD Workshop dadar', 'Thakare natya',
           'Vikhroli', 'Worli', 'vileparle W']

DISTRIBUTOR_NODE_IP_ADDR = "127.0.0.1"
DISTRIBUTOR_NODE_PORT = 5001

distributor_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
distributor_sock.bind((DISTRIBUTOR_NODE_IP_ADDR, DISTRIBUTOR_NODE_PORT))
distributor_sock.listen()

# Receive input from distributor node about the to-be-affected regions
# We are observing 36 regions, so input is a 36-dimensional vector
# An entry of 1 in the vector denotes that the region is likely to be flooded, whereas 0 denotes that the region is unlikely to be flooded
while True:
    try:
        distributor_conn, distributor_addr = distributor_sock.accept()
        print(f"Connected to distributor node: {distributor_addr}")
        while True:
            data = distributor_conn.recv(1024).decode()
            if not data:
                break
            print("Received from distributor node:", data)
            print("Each number (0/1) corresponds to a particular region")
            print("0 denotes normal rainfall forecasted in next 15 minutes")
            print("1 denotes heavy rainfall forecasted in next 15 minutes")
            status = data.split(" ")
            print()
            print("Decoding and sending commands to respective base stations")
            print()
            assert(len(status) == len(regions))
            for i in range(len(regions)):
                region = regions[i]
                base_stations = get_BS_by_region(region)
                pws_servers = get_PWS_IWF_servers(region)
                if status[i] == "1":
                    send_e2(region, base_stations, 1, cmds["HIGH_PRIORITY_RESOURCE_ALLOCATION"])
                    send_e2_pws(region, pws_servers, 1, cmds["BROADCAST_MESSAGE"])
                else:
                    send_e2(region, base_stations, 0, [cmds["NORMAL_RESOURCE_ALLOCATION"]])
    except:
        distributor_sock.close()
        break


