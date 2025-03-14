from datetime import datetime # datetime library for manipulating dates and times
import numpy as np  # NumPy library for numerical operations
import pandas as pd # Pandas library for data manipulation
import mediapipe as mp  # MediaPipe library for machine learning solutions


# Define function for calculating Euclidean distance (stride length)
def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    p1 (tuple): The (x, y) coordinates of the first point.
    p2 (tuple): The (x, y) coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    # Calculate the difference in x-coordinates and y-coordinates
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    
    # Calculate the Euclidean distance using the Pythagorean theorem
    distance = (dx**2 + dy**2) ** 0.5
    
    return distance


def calculate_angle(a, b, c):
    """
    Calculate angle between three points (a, b, c).
    a, b, c are tuples of (x, y) coordinates.
    """
    # Vector from point a to point b
    ab = np.array([a[0] - b[0], a[1] - b[1]])
    # Vector from point c to point b
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    # Calculate the cosine of the angle using dot product and magnitudes of vectors
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    # Calculate the angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function to calculate knee angle
def calculate_knee_angle(landmarks):
    """
    Calculate the knee angle based on the positions of the hip, knee, and ankle landmarks.

    Parameters:
    landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks.

    Returns:
    float: The calculated knee angle in degrees.
    """
    # Extract x and y coordinates of the hip
    hip = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y)
    
    # Extract x and y coordinates of the knee
    knee = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y)
    
    # Extract x and y coordinates of the ankle
    ankle = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y)
    
    # Calculate the angle between the hip, knee, and ankle
    angle = calculate_angle(hip, knee, ankle)
    
    return angle

# Function to calculate stride length
def calculate_stride_length(landmarks):
    """
    Calculate the stride length based on the positions of the left and right ankles.

    Parameters:
    landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks.

    Returns:
    float: The calculated stride length.
    """

    # Extract x and y coordinates of the left ankle
    left_foot = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y)
    
    # Extract x and y coordinates of the right ankle
    right_foot = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y)
    
    # Calculate the Euclidean distance between the left and right ankle positions
    stride_length = euclidean_distance(left_foot, right_foot)
    
    return stride_length

# Function to calculate hip stability
def calculate_hip_stability(landmarks):
    """
    Calculate the stability of the hips based on the Euclidean distance between the left and right hip landmarks.

    Parameters:
    landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks.

    Returns:
    float: The calculated hip stability.
    """

    # Extract x and y coordinates of the left hip
    left_hip = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y)
    
    # Extract x and y coordinates of the right hip
    right_hip = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y)
    
    # Calculate the Euclidean distance between the left and right hip positions
    hip_stability = euclidean_distance(left_hip, right_hip)
    
    return hip_stability

# Function to calculate forward lean angle
def calculate_forward_lean(landmarks):
    """
    Calculate the forward lean angle based on shoulder, torso, and knee landmarks.

    Parameters:
    landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks.

    Returns:
    float: The calculated forward lean angle in degrees.
    """
    
    # Extract x and y coordinates of the left shoulder
    shoulder = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y)
    
    # Extract x and y coordinates of the left hip (torso)
    torso = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y)
    
    # Extract x and y coordinates of the left knee
    knee = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y)
    
    # Calculate the angle between the shoulder, torso, and knee
    angle = calculate_angle(shoulder, torso, knee)
    
    return angle

# Function to calculate stride cadence (steps per second)
def calculate_stride_cadence(pose_estimates, frame_time):
    """
    Calculate the stride cadence (steps per second) from pose estimates.

    Parameters:
    pose_estimates (list): List of tuples containing frame name and pose landmarks.
    frame_time (float): Time duration of each frame.

    Returns:
    float: Stride cadence (steps per second).
    """
    
    left_step_count, right_step_count = 0, 0  # Initialize step counts for left and right foot
    prev_left_foot, prev_right_foot = None, None  # Initialize previous positions of left and right foot

    for frame_name, landmarks in pose_estimates:
        
        left_foot = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y)  # Get left foot position
        right_foot = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y)  # Get right foot position

        if prev_left_foot is not None and prev_right_foot is not None:
            # Detect if a step has occurred for left and right foot
            
            left_step, right_step, _, _ = detect_stride_cadence(landmarks, prev_left_foot, prev_right_foot, frame_time)
            if left_step:
                left_step_count += 1  # Increment left step count if a step is detected
            if right_step:
                right_step_count += 1  # Increment right step count if a step is detected
        
        prev_left_foot, prev_right_foot = left_foot, right_foot  # Update previous foot positions

    total_steps = left_step_count + right_step_count  # Calculate total steps

    return total_steps / (len(pose_estimates) * frame_time)  # Calculate and return stride cadence (steps per second)

# Function to calculate arm swing angle
def calculate_arm_swing(landmarks):
    """
    Calculate the arm swing angle based on shoulder and hip landmarks.

    Parameters:
    landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks.

    Returns:
    float: The calculated arm swing angle in degrees.
    """
     
    # Extract x and y coordinates of the left shoulder
    left_shoulder = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y)
    
    # Extract x and y coordinates of the right shoulder
    right_shoulder = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y)
    
    # Extract x and y coordinates of the left hip (torso)
    torso = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y)
    
    # Calculate the angle between the left shoulder, right shoulder, and torso
    angle = calculate_angle(left_shoulder, right_shoulder, torso)
    return angle

# Function to get position (left_hip position as centre of mass)
def get_position(landmarks):
    """
    Get the position of the left hip from the pose landmarks.

    Parameters:
    landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks.

    Returns:
    tuple: Position of the left hip as (x, y) coordinates.

    Raises:
    ValueError: If the landmark positions are negative.
    """
    
    # Extract x and y coordinates of the left hip
    left_hip_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x
    left_hip_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y
    
    # Check if the coordinates are non-negative
    if left_hip_x < 0 or left_hip_y < 0:
        raise ValueError("Landmark positions must be non-negative")
    
    # Return the position as a tuple
    return (left_hip_x, left_hip_y)

# Function to compute velocity (current position - previous position / frame time)
def compute_velocity(current_position, prev_position, frame_time):
    """
    Compute the velocity given the current and previous positions and the frame time.

    Parameters:
    current_position (tuple): Current position as a tuple (x, y).
    prev_position (tuple): Previous position as a tuple (x, y).
    frame_time (float): Time duration of each frame.

    Returns:
    float: Combined velocity.
    """
    if frame_time == 0:
        raise ValueError("Frame time must be greater than zero.")  # Raise an error if frame time is zero to avoid division by zero

    # Calculate velocity for each component (x and y)
    vx = (current_position[0] - prev_position[0]) / frame_time
    vy = (current_position[1] - prev_position[1]) / frame_time

    # Calculate combined velocity using Euclidean distance formula
    combined_velocity = (vx**2 + vy**2) ** 0.5

    return combined_velocity  # Return the velocity 

# Function to compute acceleration (current velocity - previous velocity / frame time)
def compute_acceleration(current_velocity, prev_velocity, frame_time):
    """
    Compute the acceleration given the current and previous velocities and the frame time.

    Parameters:
    current_velocity (float): Current velocity (float)
    prev_velocity (float): Previous velocity (float)
    frame_time (float): Time duration of each frame.

    Returns:
    float: Combined acceleration.
    """
    if frame_time == 0:
        raise ValueError("Frame time cannot be zero.")  # Raise an error if frame time is zero to avoid division by zero

    # Calculate acceleration
    acceleration = (current_velocity - prev_velocity) / frame_time  # Dividing by frame time instead of previous velocity
    
    return acceleration  # Return the acceleration

# Function to calculate specified metrics and append to data frame
def calculate_metrics_for_frame(landmarks, frame_time):
    """
    Calculate all metrics from pose landmarks for a single frame and return as a DataFrame.

    Args:
        landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks for the frame.
        frame_time (float): Time duration of the frame.

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics for the frame.
    """
    print("Calculating knee angle...")
    # Calculate knee angle
    knee_angle = calculate_knee_angle(landmarks)
    print(f"Knee angle: {knee_angle}")

    print("Calculating stride length...")
    # Calculate stride length
    stride_length = calculate_stride_length(landmarks)
    print(f"Stride length: {stride_length}")

    print("Calculating hip stability...")
    # Calculate hip stability
    hip_stability = calculate_hip_stability(landmarks)
    print(f"Hip stability: {hip_stability}")

    print("Calculating forward lean angle...")
    # Calculate forward lean angle
    lean_angle = calculate_forward_lean(landmarks)
    print(f"Forward lean angle: {lean_angle}")

    print("Calculating arm swing angle...")
    # Calculate arm swing angle
    arm_swing_angle = calculate_arm_swing(landmarks)
    print(f"Arm swing angle: {arm_swing_angle}")

    print("Getting position...")
    # Calculate position
    position = get_position(landmarks)
    print(f"Position: {position}")

    # Create a dictionary to store the data
    data = {
        "knee_angle": knee_angle,
        "stride_length": stride_length,
        "hip_stability": hip_stability,
        "lean_angle": lean_angle,
        "arm_swing_angle": arm_swing_angle,
        "position": position
    }

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame([data])
    
    return df


# Example usage
def get_video_fps(video_path):
    """
    Get the frames per second (FPS) of a video.

    Parameters:
    video_path (str): Path to the video file.

    Returns:
    float: Frames per second (FPS) of the video.

    Raises:
    ValueError: If FPS could not be determined.
    """
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    # Retrieve the FPS property from the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    cap.release()
    
    # Check if FPS is valid
    if fps == 0:
        raise ValueError("FPS could not be determined. Please check the video path or file.")
    
    return fps


# Function to track foot movements over time
def detect_stride_cadence(landmarks, prev_left_foot, prev_right_foot, frame_time):
    """
    Detect stride cadence by tracking foot movements over time.

    Parameters:
    landmarks (google._upb._message.RepeatedCompositeContainer): Pose landmarks.
    prev_left_foot (numpy.ndarray): Previous position of the left foot.
    prev_right_foot (numpy.ndarray): Previous position of the right foot.
    frame_time (float): Time duration of each frame.

    Returns:
    tuple: A tuple containing boolean values indicating left and right steps, and the current positions of left and right feet.
    """
    
    # Extract current positions of left and right feet
    left_foot = np.array([landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y])
    right_foot = np.array([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y])

    # Detect movement by checking the change in distance
    left_movement = np.linalg.norm(left_foot - prev_left_foot) if prev_left_foot is not None else 0
    right_movement = np.linalg.norm(right_foot - prev_right_foot) if prev_right_foot is not None else 0

    # Consider a movement significant if it's above a threshold
    movement_threshold = 0.01  # Adjust based on camera angle & resolution
    left_step = left_movement > movement_threshold
    right_step = right_movement > movement_threshold

    return left_step, right_step, left_foot, right_foot


def calculate_time_series_metrics(df_skating, pose_estimates, frame_time):
    """
    Calculate time series metrics and update the DataFrame with new columns.

    Parameters:
    df_skating (pd.DataFrame): DataFrame containing skating metrics.

    Returns:
    pd.DataFrame: Updated DataFrame with new time series metrics columns.
    """
    print("Calculating stride cadence...")
    # Calculate stride cadence
    df_skating['stride_cadence'] = calculate_stride_cadence(pose_estimates, frame_time)
    print("Stride cadence calculated.")

    #print("Calculating velocity...")
    # Calculate velocity
    df_skating['velocity'] = 0
    #df_skating.apply(lambda row: compute_velocity(row['position'], df_skating['position'].shift(), frame_time), axis=1)
    #print("Velocity calculated.")

    #print("Calculating acceleration...")
    # Calculate acceleration
    df_skating['acceleration'] = 0
    #df_skating.apply(lambda row: compute_acceleration(row['velocity'], row['velocity'].shift(), frame_time), axis=1)
    #print("Acceleration calculated.")
    
    print("Calculating velocity change...")
    # Calculate velocity change (difference between frames)
    df_skating['velocity_change'] = df_skating['velocity'].diff()
    print("Velocity change calculated.")

    print("Calculating acceleration change...")
    # Calculate acceleration change (difference between frames)
    df_skating['acceleration_change'] = df_skating['acceleration'].diff()
    print("Acceleration change calculated.")

    print("Calculating stride consistency...")
    # Calculate stride consistency (rolling mean of 5 stride lengths)
    df_skating['stride_consistency'] = df_skating['stride_length'].rolling(window=5).mean()
    print("Stride consistency calculated.")

    print("Calculating cadence stability...")
    # Calculate cadence stability (rolling standard deviation of 10 stride cadence)
    df_skating['cadence_stability'] = df_skating['stride_cadence'].rolling(window=10).std()
    print("Cadence stability calculated.")

    print("Calculating knee stability...")
    # Calculate knee stability (rolling variance of 5 knee angles)
    df_skating['knee_stability'] = df_skating['knee_angle'].rolling(window=5).var()
    print("Knee stability calculated.")

    print("Handling missing values...")
    # Handle Missing Values (from Rolling Window Operations)
    df_skating.fillna(0, inplace=True)  # Fill NaN values with 0
    print("Missing values handled.")

    return df_skating

def get_feedback_for_prediction(frame_predictions):
    """
    Get feedback for a list of predictions.

    Parameters:
    predictions (list): List of predictions, each containing 8 label codes per frame.

    Returns:
    list: List of feedback strings for each frame.
    """
    feedback_list = []

    # Define feedback labels and codes
    feedback_labels = {
        "knee angle": {
            0: "Try to bend your knees more",
            1: "Keep practicing",
            2: "Great job!"
        },
        "stride length": {
            0: "Increase your stride length",
            1: "Keep practicing",
            2: "Great job!"
        },
        "stride cadence": {
            0: "Increase your cadence",
            1: "Keep practicing",
            2: "Great job!"
        },
        "hip stability": {
            0: "Improve your hip stability",
            1: "Keep practicing",
            2: "Great job!"
        },
        "forward lean": {
            0: "Lean forward more",
            1: "Keep practicing",
            2: "Great job!"
        },
        "arm swing": {
            0: "Improve your arm swing",
            1: "Keep practicing",
            2: "Great job!"
        },
        "velocity": {
            0: "Increase your speed",
            1: "Keep practicing",
            2: "Great job!"
        },
        "acceleration": {
            0: "Improve your acceleration",
            1: "Keep practicing",
            2: "Great job!"
        }
    }

    # List of metric names in order
    metric_names = [
        "knee angle",
        "stride length",
        "stride cadence",
        "hip stability",
        "forward lean",
        "arm swing",
        "velocity",
        "acceleration"
    ]

    print("Processing predictions...")

    for i in range(len(frame_predictions)):  # Iterate over all label codes in frame
        predicted_label_code = frame_predictions[i]
        metric_name = metric_names[i]  # Get metric name

        # Ensure metric exists in feedback_labels and get feedback
        if metric_name in feedback_labels and predicted_label_code in feedback_labels[metric_name]:
            feedback_list.append(f"{metric_name.capitalize()}: {feedback_labels[metric_name][predicted_label_code]}")
        else:
            feedback_list.append(f"{metric_name.capitalize()}: Unknown feedback")  # Default case

    return feedback_list
    