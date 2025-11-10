################################################################################
# filename: creating_files.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 10/11,2025
################################################################################

import os
import json
import socket, threading

################################################################################

HOST, PORT = "127.0.0.1", 5000
latest_state = None

################################################################################

def create_file(path_to_file : str, datas : dict):
    with open(path_to_file, "w") as file:
        json.dump(datas, file)
    
################################################################################

def listener():
    global latest_state
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            conn, _ = s.accept()
            with conn:
                data = conn.recv(65536)
                if not data:
                    continue
                latest_state = json.loads(data.decode('utf-8'))

################################################################################

def main():
    print("Le code se lance et attend les informations")
    iteration = 0
    while True:
        if latest_state is not None:
            state = latest_state
            latest_state = None
            iteration += 1
            if not os.path.exists(f"data_recognize/raw.json"):
                os.mkdir(f"data_recognize/raw", exist_ok=True)
            create_file(f"data_recognize/raw/datas_{iteration}.json", state)

################################################################################

if __name__ == "__main__":
    main()

################################################################################
# End of File
################################################################################