import cv2
import socket
import struct
import threading
import numpy as np
from time import time
from ultralytics import YOLO, checks

# Global variable to store the model and raspberry object
model = None
raspberry = None

# Constants
AI_CONFIDANCE_THRESHOLD = 0.5
HOSTNAME = "192.168.1.100"
PORT_CAMERA = 3131
PORT_MOTOR = 6969

# Image and Camera Parameters
IMAGE_FOLDER = "bilkentorin/input"  # Input folder
OUTPUT_FOLDER = "bilkentorin/output"  # Output folder

HORIZONTAL_PIXELS_WIDE_CAM = 2560
VERTICAL_PIXELS_WIDE_CAM = 1440
HORIZONTAL_PIXELS_NARROW_CAM = 640
VERTICAL_PIXELS_NARROW_CAM = 480
FPS_WIDE_CAM = 30  # Frames per second for the wide camera (THIS CAN BE EXPERIMENTALLY DETERMINED)
FPS_NARROW_CAM = 15  # Frames per second for the narrow camera
FOV_HORIZONTAL_WIDE_CAM = 60  # Field of View of wide cam in degrees (Horizontal)
FOV_VERTICAL_WIDE_CAM = 33.75  # Field of View of wide cam in degrees (Vertical)
FOV_HORIZONTAL_NARROW_CAM = 60  # Field of View of narrow cam in degrees (Horizontal)
FOV_VERTICAL_NARROW_CAM = 45  # Field of View of narrow cam in degrees (Vertical)
CAMERA_TILT_ANGLE = 5 + FOV_VERTICAL_WIDE_CAM / 2 # Tilt angle of the wide angle camera
IMAGE_EXTENSION = ".jpg"  # Image file extension
SCAN_ANGLE = 350  # The angle of the scan in degrees
MOTOR_SCAN_ANGLE = SCAN_ANGLE - FOV_HORIZONTAL_WIDE_CAM  # The motor scan angle in degrees
SCAN_PERIOD = 3  # The period of the scan in seconds


class RaspberryConnection:
    def __init__(self, host : str, ports : int):
        self.server_address_camera = (host, ports[0])
        self.client_socket_camera = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_camera.connect(self.server_address_camera)
        print(f"Connected to Raspberry Pi for Camera {host}:{ports[0]}")
        self.server_address_motor = (host, ports[1])
        self.client_socket_motor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_motor.connect(self.server_address_motor)
        print(f"Connected to Raspberry Pi for Motor {host}:{ports[1]}")

    def captureFrameNarrow(self):
        """
        :param 
        None
        
        :return 
        frame: the frame captured from the narrow angle camera

        description: this function will capture the frame from the narrow camera
        """
        self.client_socket_camera.sendall(b'REQF')

        data_size = struct.unpack("L", self.client_socket_camera.recv(8))[0]
        data = b""
        while len(data) < data_size:
            packet = self.client_socket_camera.recv(4096)
            if not packet:
                break
            data += packet

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame
    
    @staticmethod
    def captureFrameWide():
        """
        :param 
        None
        
        :return 
        frame: the frame captured from the wide angle camera

        description: this function will capture the frame from the wide camera
        """
        wide_cam = cv2.VideoCapture(0)
        ret, frame = wide_cam.read()
        if not ret:
            print("Error: Could not capture frame from wide camera.")
        wide_cam.release()
        return frame
    
    def motorControlNarrow(self, initial_angles : tuple, desired_angles : tuple, t : float):
        """
        :param
        pin: the pin number of the motor
        initial_angle: the initial angle of the motor
        desired_angle: the desired angle of the motor
        t: the time it takes to turn the motor to the desired angle

        :return 
        None

        description: this function will turn the motors to the desired 
        angle in given duration t using ethernet with RP wait until the motor is turned
        """
        pass

    def motorControlWide(self, initial_angle : float, desired_angle : float, t : float):
        """
        :param
        pin: the pin number of the motor
        initial_angle: the initial angle of the motor
        desired_angle: the desired angle of the motor
        t: the time it takes to turn the motor to the desired angle

        :return 
        None

        description: this function will turn the motor to the desired 
        angle in given duration t using ethernet with RP wait until the motor is turned
        """
        pass

    

    def close(self):
        self.client_socket_camera.close()
        self.client_socket_motor.close()

class DroneCandidate:
    ID = 0  # Class variable for unique ID generation
    MAX_LOST_TIME = 1.0  # Maximum time (in seconds) to consider a drone candidate lost
    # Measurement matrix: we only measure azimuth and elevation.
    H = np.array([
        [1, 0, 0, 0],  # Measurement for azimuth
        [0, 0, 1, 0]   # Measurement for elevation
    ])
    
    # Process noise covariance (4x4) for the constant velocity model.
    Q = np.eye(4) * 0.1
    
    # Measurement noise covariance (2x2)
    R = np.eye(2) * 0.01

    @staticmethod
    def F(dt):
        """
        State transition matrix for a constant velocity model.
        
        The state vector is defined as:
          x = [azimuth, angular_velocity, elevation, elevation_velocity]
        
        So, the state update is:
          azimuth_new = azimuth + angular_velocity * dt
          angular_velocity_new = angular_velocity  (constant velocity)
          elevation_new = elevation + elevation_velocity * dt
          elevation_velocity_new = elevation_velocity  (constant velocity)
        """
        return np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ])

    def __init__(self, alpha : tuple, t : float):
        """
        Initialize the drone candidate.
        
        Parameters:
          alpha: list or array-like with [azimuth, elevation] in degrees.
          t:     time (in seconds) of the observation.
        """
        self.alpha = np.array(alpha)  # Observed [azimuth, elevation]
        self.t = t  # Time of last observation
        
        # State vector: [azimuth, angular_velocity, elevation, elevation_velocity]
        # Initially, we set the angular velocities to zero.
        self.x = np.array([alpha[0], 0, alpha[1], 0])
        
        # Covariance matrix (uncertainty of the state)
        self.P = np.eye(4) * 0.5

        self.id = DroneCandidate.ID # Unique ID for tracking
        DroneCandidate.ID += 1 # Increment the ID counter
        
    def KalmanPredict(self, t_desired : float):
        """
        Predict the next state at a desired time using the constant velocity model.
        
        Parameters:
          t_desired: The time at which to predict the state.
        
        Returns:
          A tuple (predicted_azimuth, predicted_elevation)
        """
        dt = t_desired - self.t
        F = self.F(dt)
        
        # Predict state and covariance.
        self.x = F @ self.x
        
        # Return the predicted azimuth and elevation (wrap azimuth if needed).
        return self.x[0] % 360, (self.x[2] + 90) % 180 - 90

    def UpdateObservation(self, alpha : tuple, t : float):
        """
        Update the state with a new observation using Kalman filtering.
        
        Parameters:
          alpha: New measurement [azimuth, elevation]
          t:     Time of the new observation.
        """
        dt = t - self.t
        F = self.F(dt)
        
        # First, predict the state to the current time.
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
        # New measurement.
        Z = np.array(alpha)
        # Innovation (measurement residual)
        y = Z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate and covariance.
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # Update the stored measurement and time.
        self.alpha = np.array([self.x[0], self.x[2]])
        self.t = t


def convertPolarWide(raw_candidates, motor_angle):
    """
    :param
    raw_candidates: the candidates from the non-AI algorithm
    motor_angle: the last motor angle
    :return
    raw_candidates_polar: the candidates in polar coordinates
    description: this function will take the raw_candidates and convert
    them into polar coordinates from pixel values using last motor angle 
    """
 
    # Calculate focal lengths in pixels from FOV values
    f_x = (HORIZONTAL_PIXELS_WIDE_CAM / 2) / np.tan(np.radians(FOV_HORIZONTAL_WIDE_CAM) / 2)
    f_y = (VERTICAL_PIXELS_WIDE_CAM / 2) / np.tan(np.radians(FOV_VERTICAL_WIDE_CAM) / 2)
    
    # Assume the principal point is at the center of the image
    center_x = HORIZONTAL_PIXELS_WIDE_CAM / 2.0
    center_y = VERTICAL_PIXELS_WIDE_CAM / 2.0

    raw_candidates_polar = []
    
    for candidate in raw_candidates:
        u, v = candidate  # pixel coordinates

        # Compute displacement from the image center
        x = u - center_x
        y = v - center_y

        # Calculate angles (in radians) using arctan
        angle_x_rad = np.arctan(x / f_x)
        angle_y_rad = np.arctan(y / f_y)
        
        # Convert radians to degrees
        angle_x_deg = np.degrees(angle_x_rad)
        angle_y_deg = np.degrees(angle_y_rad)

        # Adjust with the motor angle and camera tilt
        horizontal_angle = (motor_angle + angle_x_deg) % 360
        vertical_angle = (angle_y_deg + CAMERA_TILT_ANGLE + 90) % 180 - 90

        raw_candidates_polar.append([horizontal_angle, vertical_angle])
    
    return raw_candidates_polar

def convertPolarNarrow(detected_drones, motor_angles):
    """
    :param
    detected_drones: the candidates from the AI algorithm
    motor_angles: the last motor angles, theta and phi
    :return
    detected_drones_polar: the candidates in polar coordinates
    description: this function will take the detected_drones and convert
    them into polar coordinates from pixel values using last motor angle
    """

    
    # Calculate focal lengths in pixels from FOV values
    f_x = (HORIZONTAL_PIXELS_NARROW_CAM / 2) / np.tan(np.radians(FOV_HORIZONTAL_NARROW_CAM) / 2)
    f_y = (VERTICAL_PIXELS_NARROW_CAM / 2) / np.tan(np.radians(FOV_VERTICAL_NARROW_CAM) / 2)
    
    # Assume the principal point is at the center of the image
    center_x = HORIZONTAL_PIXELS_NARROW_CAM / 2.0
    center_y = VERTICAL_PIXELS_NARROW_CAM / 2.0
    
    detected_drones_polar = []

    for candidate in detected_drones:
        u, v = candidate['centers']  # pixel coordinates

        # Compute displacement from the image center
        x = u - center_x
        y = v - center_y

        # Calculate angles (in radians) using arctan
        angle_x_rad = np.arctan(x / f_x)
        angle_y_rad = np.arctan(y / f_y)
        
        # Convert radians to degrees
        angle_x_deg = np.degrees(angle_x_rad)
        angle_y_deg = np.degrees(angle_y_rad)

        # Adjust with the motor angle and camera tilt
        horizontal_angle = (motor_angles[0] + angle_x_deg) % 360
        vertical_angle = ((motor_angles[1] + angle_y_deg) + 90) % 180 - 90 

        detected_drones_polar.append({
                "centers": (horizontal_angle, vertical_angle),
                "confidence": candidate["confidence"]
            })

    return detected_drones_polar


def resetMotors():
    """
    :param 
    None
    
    :return 
    None

    description: this function will reset the motors to the initial position
    """
    pass


def isButtonPressed(pin : int):
    """
    :param 
    pin: the pin number of the button
    
    :return 
    True if the button is pressed

    description: this function will check if the button is pressed
    """
    pass

def initialize_model():
    """
    Initializes and loads the trained YOLO model.
    """
    global model
    checks()
    model = YOLO("best.pt")  # Load the trained YOLO model

def initialiseAllSystems():
    """
    :param 
    None
    
    :return 
    None

    description: this function will initialise the buttons motors yolo and world
    """
    global HOSTNAME
    global PORT_CAMERA
    global PORT_MOTOR
    global raspberry

    raspberry = RaspberryConnection(HOSTNAME, (PORT_CAMERA, PORT_MOTOR))
    initialize_model()
    resetMotors()

def LEDScreenShow(droneCandidateList):
    """
    :param
    droneCandidateList: the list of drone candidates
    :return
    None
    description: this function will show the drones via LED screen
    """
    pass

def dataAssociation(raw_candidates, t):
    """Updates tracked drones with new detections, keeping unique IDs for tracking."""
    global droneCandidateList  # Use global counter for unique IDs
    global undeterminedDroneCandidateList
    global Lock

    for alpha in raw_candidates:
        matched = False        
        # Try to associate detection with an existing tracked drone
        for drone in droneCandidateList:
            predicted_azimuth, predicted_elevation = drone.KalmanPredict(t)
           
            # Check if the detection is close enough to the predicted state (angle is in degrees)
            if (t - drone.t < DroneCandidate.MAX_LOST_TIME):
                if abs((predicted_azimuth - alpha[0] + 180) % 360 - 180) < 6.5 and abs(predicted_elevation - alpha[1]) < 6.5:
                    # this part can be improved by checking other drones and picking the closest one 
                    # !!!DO THE IMPROVEMENT!!!
                    drone.UpdateObservation(alpha, t)
                    matched = True
                    break  # Stop checking after first 
            # This part can be reconsidered with Ahmed since we do not want him to meow
            else:
                 if abs((drone.alpha[0] - alpha[0] + 180) % 360 - 180) < 10 and abs(drone.alpha[1] - alpha[1]) < 10:
                      drone.UpdateObservation(alpha, t)
                      matched = True
                      break  # Stop checking after first 
                    
        # If no match is found, register a new drone with a unique ID     
        if not matched:
            new_drone = DroneCandidate(alpha, t)
            droneCandidateList.append(new_drone)

    # Identify stale drones (if they haven't been updated for max_lost_time)
    expired_drones = [drone for drone in droneCandidateList if (t - drone.t) >= DroneCandidate.MAX_LOST_TIME]
    # Remove stale drones from the tracking list
    droneCandidateList[:] = [drone for drone in droneCandidateList if (t - drone.t) < DroneCandidate.MAX_LOST_TIME]
    # Append the dronce candidates that have .isDrone flag None into the undeterminedDroneCandidateList but also check that this object is not previously appended
    for drone in droneCandidateList:
        if drone.isDrone == None and drone not in undeterminedDroneCandidateList:
            with Lock:
                undeterminedDroneCandidateList.append(drone)

    # Explicitly delete expired drone objects
    with Lock:
        for drone in expired_drones:
            del drone