from utils import *

def run_inference(frame):
    """
    Runs inference on the given frame using the YOLO model.
    
    :param 
    frame: The input image frame from the narrow-angle camera.

    :return: List of detected drones with their bounding boxes and confidence scores.
    """
    global model

    if model is None:
        raise ValueError("Model is not initialized. Call initialize_model() first.")

    results = model(frame)  # Run YOLO inference
    detected_drones = []

    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            # change the next line according to tested code
            x_min, y_min, x_max, y_max = map(int, box.xyx.cpu().numpy()) # make this correct it is incorrect now
            confidence = box.conf[0].item()

            # Assuming YOLO is trained only to detect drones, we don't need class filtering.
            detected_drones.append({
                "centers": ((x_min+x_max)/2, (y_min+y_max)/2),
                "confidence": confidence
            })

    return detected_drones



def AIConfirmation(droneCandidate : DroneCandidate, last_motor_angles : tuple):
    """
    :param
    droneCandidate: the drone candidate to be confirmed
    last_motor_angles: the last motor angles to be used for motorControl function

    :return 
    last_motor_angles: the last motor angles of the motor
    
    description: this function will take the drone candidate and confirm it 
    using the AI camera and change the isDrone flag of the drone candidate
    lastly it will return the last positions of the motor angles
    """
    global Lock
    global raspberry

    # get the motor angles using droneCandidate .KalmanPredict function this will return the position of the drone
    predicted_azimuth, predicted_elevation = droneCandidate.KalmanPredict(time())

    # turn the motors to the predicted angles wait until the motors are turned
    raspberry.motorControlNarrow(last_motor_angles, (predicted_azimuth, predicted_elevation), 1/FPS_NARROW_CAM*0.9)

    # capture the frame from the narrow camera
    last_seen_time = time()
    frame = raspberry.captureFrameNarrow()

    # run the inference on the frame
    # (detected drones is a list of dictionaries each dictionary contains the center of the detected drone and the confidence of the detection)
    detected_drones = run_inference(frame)

    # convert the pixel coordinates to the angles
    detected_drones = convertPolarNarrow(detected_drones, (predicted_azimuth, predicted_elevation))

    # if there are more than one drone detected then get the closest drone to the predicted position
    if len(detected_drones) > 0:
        if len(detected_drones) > 1:
            # get the closest drone to the predicted position not in a function
            smallest_distance = np.inf
            closest_drone = None
            for detected_drone in detected_drones:
                # get the distance between the predicted position and the detected drone
                distance = np.sqrt(((predicted_azimuth - detected_drone["centers"][0] + 180) % 360 - 180)**2 + (predicted_elevation - detected_drone["centers"][1])**2)
                # if the distance is smaller than the smallest distance then update the smallest distance and the detected drone
                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_drone = detected_drone
        else:
            # if there is only one drone detected then get the drone
            closest_drone = detected_drones[0]

        # if the confidence of the closest drone is greater than 0.5 then update the isDrone flag of the drone candidate
        if closest_drone["confidence"] > AI_CONFIDANCE_THRESHOLD:
            with Lock:
                droneCandidate.isDrone = True
                droneCandidate.UpdateObservation(closest_drone["centers"], last_seen_time)
        else:
            # if the confidence is less than 0.5 then update the isDrone flag of the drone candidate
            with Lock:
                droneCandidate.isDrone = False
    else:
        # if there is no drone detected then update the isDrone flag of the drone candidate
        with Lock:
            droneCandidate.isDrone = False

    # return the motor angles
    return (predicted_azimuth, predicted_elevation)