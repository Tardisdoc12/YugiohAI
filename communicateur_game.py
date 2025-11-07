import socket, json, threading

HOST, PORT = "127.0.0.1", 5000
latest_state = None

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


threading.Thread(target=listener, daemon=True).start()

print("Le code se lance et attend les informations")
while True:
    if latest_state is not None:
        state = latest_state
        latest_state = None
        print("Processing latest state:", state)
        #suite du traitement