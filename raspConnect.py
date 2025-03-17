import cv2
import time
import socket
import struct
import pickle
import threading
import RPi.GPIO as GPIO # type: ignore

HOSTNAME = '192.168.1.100'
CAMERA_PORT = 1234
MOTOR_PORT = 3131

# Define GPIO pins for motors
# Motor pins for wide angle camera
wide_angle_step_pin = 17  # Wide angle step pin
wide_angle_dir_pin = 22   # Wide angle direction pin

# Narrow angle camera horizontal motor pins
narrow_angle_Hstep_pin = 18  # Narrow angle horizontal step pin
narrow_angle_Hdir_pin = 23   # Narrow angle horizontal direction pin

# Narrow angle camera vertical motor pins
narrow_angle_Vstep_pin = 27  # Narrow angle vertical step pin
narrow_angle_Vdir_pin = 24   # Narrow angle vertical direction pin

# Initialize GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(wide_angle_step_pin, GPIO.OUT)
GPIO.setup(wide_angle_dir_pin, GPIO.OUT)

GPIO.setup(narrow_angle_Hstep_pin, GPIO.OUT)
GPIO.setup(narrow_angle_Hdir_pin, GPIO.OUT)

GPIO.setup(narrow_angle_Vstep_pin, GPIO.OUT)
GPIO.setup(narrow_angle_Vdir_pin, GPIO.OUT)


def send_frame(conn):
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)  # Open camera
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not capture frame.")
        return

    # Compress frame to JPEG
    _, buffer = cv2.imencode(".jpg", frame)
    data = buffer.tobytes()

    # Send data length first
    conn.sendall(struct.pack("L", len(data)))

    # Send frame data
    conn.sendall(data)

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOSTNAME, CAMERA_PORT))
    server_socket.listen(1)
    print(f"Server listening on {HOSTNAME}:{CAMERA_PORT}")

    conn, addr = server_socket.accept()
    print(f"Connection established with {addr}")

    while True:
        request = conn.recv(4)
        if not request:
            break

        if request == b'REQF':  # Frame request
            send_frame(conn)
    conn.close()

# motor control function
def rotate_motor(initial_angle, desired_angle, step_pin, direction_pin, t):
    not_sleep = 0
    sleep_time_begin = time.time()
    # Calculate movement per step (assuming 360 degree rotation requires 200 steps)
    steps_per_rev = 200  # required steps for 360 degree rotation
    step_angle = 360 / steps_per_rev  # angle per step

    # angle difference
    angle_diff = abs(desired_angle - initial_angle)

    # Determine the direction of rotation
    if desired_angle > initial_angle:
        direction = GPIO.HIGH  # clockwise rotation
        print("Direction High, clockwise rotation")
    else:
        direction = GPIO.LOW   # counter-clockwise rotation
        print("Direction Low, counter-clockwise rotation")

    # establish the direction
    GPIO.output(direction_pin, direction)

    # calculate the number of steps to move
    steps_to_move = int(angle_diff / step_angle)  # number of steps to move

    # calculate the delay between steps
    step_delay = (0.5*t) / steps_to_move  # delay between steps
    
    for _ in range(steps_to_move):
        t1 = time.time()
        GPIO.output(step_pin, GPIO.HIGH)
        t2 = time.time()
        not_sleep += t2-t1
        time.sleep(step_delay)
        t3 = time.time()
        GPIO.output(step_pin, GPIO.LOW)
        t4 = time.time()
        not_sleep += t4-t3
        time.sleep(step_delay)

    print(f"{initial_angle}° -> {desired_angle}° with {steps_to_move} steps.")
    sleep_time_total = time.time() - sleep_time_begin
    print(f"Total time for turn: {sleep_time_total:.2f}")

# listener for motor control commands
def start_listener():

    server_address = (HOSTNAME, MOTOR_PORT) # server address
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(server_address)
        server_socket.listen(1)
        print(f"Listening {server_address}...")

        connection, client_address = server_socket.accept()
        print(f"Connection established: {client_address}")
        
        while True:
            data = connection.recv(1024)  # get data
            if data:
                message = pickle.loads(data)  # Unpickle the data
                print(f"Command: {message}")
                initial_angle, desired_angle, t =  message
                if isinstance(initial_angle, tuple):  # narrow angle motor
                    # narrow angle motor (horizontal and vertical)
                    initial_angle_x, initial_angle_y = initial_angle
                    desired_angle_x, desired_angle_y = desired_angle

                    # horizontal motor
                    narrow_h = threading.Thread(target=rotate_motor, args=(initial_angle_x, desired_angle_x, narrow_angle_Hstep_pin, narrow_angle_Hdir_pin, t))

                    # vertical motor
                    narrow_v = threading.Thread(target=rotate_motor, args=(initial_angle_y, desired_angle_y, narrow_angle_Vstep_pin, narrow_angle_Vdir_pin,t))
                    
                    # start and join the threads
                    narrow_h.start()
                    narrow_v.start()
                    narrow_h.join()
                    narrow_v.join()
                    
                else:  # Wide angle motor
                    rotate_motor(initial_angle, desired_angle, wide_angle_step_pin, wide_angle_dir_pin, t)

if __name__ == "__main__":
    try:
        cam_thread = threading.Thread(target=start_server)
        motor_thread = threading.Thread(target=start_listener)

        cam_thread.start()
        motor_thread.start()
        
    except KeyboardInterrupt or OSError:
        print("\nExiting...")
        GPIO.cleanup()  # Clean up GPIO pins
