from utils import *

def nonAIScan():
    """
    This function is used to scan for drones using non-AI methods.
    The function will capture frames from the wide angle camera and
    process the frames using non-AI methods to detect drones.
    Also, the function rotates the motor to scan the area.
    """
    global raspberry
    global startCode

    # Calculate the step size
    step_size = 2 * MOTOR_SCAN_ANGLE / (FPS_WIDE_CAM * SCAN_PERIOD) # dividing FPS_WIDE_CAM might be unnecasarry due to cv2 implementation of VideoCapture

    # Generate the forward sequence from step_size up to determined_angle (excluding)
    forward = np.arange(step_size, MOTOR_SCAN_ANGLE + step_size - MOTOR_SCAN_ANGLE % step_size, step_size)

    # Generate the backward sequence from determined_angle (or more, including) down to 0 (including)
    backward = np.arange(MOTOR_SCAN_ANGLE + step_size - MOTOR_SCAN_ANGLE % step_size, -1, -step_size)

    # Concatenate to get the motor lookup table
    # It goes like (5, 10, 15, ..., 90, 85, 80, ..., 5, 0) if 89 is the scan angle
    motor_angles_lookup = np.concatenate((forward, backward))

    desired_angle_index = 0 # Index of the desired angle in the motor lookup table
    current_angle_index = -1 # Index of the current angle in the motor lookup table


    while startCode:

        # Save the time when the frame is captured
        nonAIFrameTime = time()

        # Capture non-AI frame from wide angle camera
        # .captureFrameWide() is a static function, nothing to do with raspberry object
        nonAIFrame = raspberry.captureFrameWide()

        # Rotate the motor to the desired angle, continue when finished
        raspberry.motorControlWide(motor_angles_lookup[current_angle_index],
                               motor_angles_lookup[desired_angle_index], 1 / FPS_WIDE_CAM * 0.9)

        # Process the frame to detect drones using non-AI methods
        raw_candidates = nonAI_Algorithm(nonAIFrame)

        # Convert drone pixel coordinates to global angle coordinates
        raw_candidates_polar = convertPolarWide(raw_candidates, motor_angles_lookup[current_angle_index])
        
        # Call data association function to associate the detected drones with the previous detections
        dataAssociation(raw_candidates_polar, nonAIFrameTime)

        desired_angle_index = (desired_angle_index + 1) % len(motor_angles_lookup)
        current_angle_index = (current_angle_index + 1) % len(motor_angles_lookup)

#Non-AI Detection
def nonAI_Algorithm(image):
    # Load image
    if image is None:
        print("Error: Could not load image.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny Edge Detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sky-like color range in BGR
    lower_sky_color = np.array([100, 100, 120])
    upper_sky_color = np.array([255, 255, 255])
    
    def sky_pixel_ratio(vicinity_region):
        mask = cv2.inRange(vicinity_region, lower_sky_color, upper_sky_color)
        sky_pixels = cv2.countNonZero(mask)
        total_pixels = vicinity_region.size / 3  # Since it's a BGR image
        return sky_pixels / total_pixels
    
    def merge_candidates(candidates, threshold=80):
        merged = []
        used = [False] * len(candidates)
        for i, (x1, y1) in enumerate(candidates):
            if used[i]:
                continue
            merged_center = [x1, y1]
            count = 1
            for j, (x2, y2) in enumerate(candidates):
                if i != j and not used[j]:
                    center_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if center_dist <= threshold:
                        merged_center[0] += x2
                        merged_center[1] += y2
                        count += 1
                        used[j] = True
            used[i] = True
            merged.append([merged_center[0] // count, merged_center[1] // count])
        return merged
    
    drone_candidates = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        aspect_ratio = w / float(h)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        length_to_area_ratio = (perimeter / area) if area > 0 else 0
        
        if (0.2 < aspect_ratio < 5 and 0.1 <= area <= 2000 and 
            0.9 >= circularity >= 0.1 and 0.10 <= length_to_area_ratio <= 10.0) or \
            (0.2 < aspect_ratio < 5.0 and 0.1 <= area <= 2000 and 
             0.3 >= circularity >= 0.1 and 8 <= length_to_area_ratio <= 30.0):
            
            # Vicinity region
            vicinity_x1 = max(x - 10, 0)
            vicinity_y1 = max(y - 10, 0)
            vicinity_x2 = min(x + w + 10, image.shape[1])
            vicinity_y2 = min(y + h + 10, image.shape[0])
            vicinity_region = image[vicinity_y1:vicinity_y2, vicinity_x1:vicinity_x2]
            
            # Check for sky-like vicinity
            sky_ratio = sky_pixel_ratio(vicinity_region)
            
            if sky_ratio > 0.5:
                center_x = x + w // 2
                center_y = y + h // 2
                drone_candidates.append([center_x, center_y])
    
    # Merge candidates
    merged_candidates = merge_candidates(drone_candidates, threshold=80)
    
    # Convert to NumPy matrix
    if merged_candidates:
        raw_candidates = np.array(merged_candidates)  # One row per merged candidate
        return raw_candidates
    else:
        raw_candidates=np.empty((0, 2))
        return raw_candidates # Empty matrix if no drones detected