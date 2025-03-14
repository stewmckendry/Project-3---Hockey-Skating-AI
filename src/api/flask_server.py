from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os

from skating_metrics import calculate_metrics_for_frame, calculate_time_series_metrics, get_feedback_for_prediction

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load trained AI model
print("Loading AI model...")
model = joblib.load("src/models/skating_ai_model.pkl")
print("Model loaded successfully.")

# Load the trained scaler so we can scale the input features like we did during training
scaler = joblib.load("src/models/scaler.pkl")

# Initialize MediaPipe Pose
print("Initializing MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
print("MediaPipe Pose initialized.")

@app.route('/analyze', methods=['POST'])
def analyze_video():
    print("Received request to analyze video.")
    
    # Receive video file from client
    file = request.files['video']
    file_path = os.path.join("/Users/liammckendry/Project3_NHL_Videos/data/sample", file.filename)
    file.save(file_path)
    print(f"Video saved to {file_path}")

    # Open video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return jsonify({"error": "Could not open video."}), 400
    
    print("Video opened successfully.")
    
    # Initialize variables to store skating metrics
    skating_metrics = pd.DataFrame()
    frame_count = 0
    landmarks_list = []
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error reading frame.")
            break
        
        # Skip every 10 frames to speed up processing
        frame_count += 1
        if frame_count % 10 != 0:
            continue

        # Get video FPS and calculate frame time
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / fps
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Extract pose landmarks and calculate skating metrics per frame
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            landmarks_list.append((frame_count, landmarks))
            frame_skating_metrics = calculate_metrics_for_frame(landmarks, frame_time)
            skating_metrics = pd.concat([skating_metrics, frame_skating_metrics], axis=0)
            print(f"Processed frame {frame_count}")

            # Save processed frame with pose drawing
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame_path = f"uploads/processed_frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            processed_frames.append(frame_path)

    # Calculate time series metrics
    skating_metrics = calculate_time_series_metrics(skating_metrics, landmarks_list, frame_time)
    print("Time series metrics calculated.")
    
    # Release video capture
    cap.release()
    print("Video processing complete.")
        
    # Convert to DataFrame and make AI predictions
    df_test = pd.DataFrame(skating_metrics, columns=['knee_angle', 'stride_length', 'stride_cadence', 'hip_stability', 'lean_angle', 'arm_swing_angle', 'acceleration'])
    features_scaled = scaler.transform(df_test)
    print("Processed Features:", features_scaled)  # Debugging: See formatted features
        
    predictions = model.predict(features_scaled)
    print("AI predictions made. Predictions:", predictions[:10])

    # Convert predictions to a JSON-friendly format
    feedback = df_test.copy()
    feedback["Predicted_Label"] = predictions.tolist()
    feedback["Feedback"] = feedback["Predicted_Label"].apply(lambda label: get_feedback_for_prediction(label))
    feedback.drop(columns=["Predicted_Label"], inplace=True)
    print("Feedback prepared for response.")

    return jsonify({
        "feedback": feedback.to_dict(orient="records"),
        "processed_frames": processed_frames
    })

@app.route('/get_frame/<frame_name>')
def get_frame(frame_name):
    """Serve the processed frame image to the client."""
    frame_path = f"../../uploads/{frame_name}"
    return send_file(frame_path, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
    print("Flask server started.")
