import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("🏒 AI Hockey Skating Analysis")

# Upload video
uploaded_file = st.file_uploader("Upload a skating video", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)

    # Save video locally
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Send video to Flask API for processing
    with open("temp_video.mp4", "rb") as f:
        files = {'video': f}
        response = requests.post("http://127.0.0.1:5001/analyze", files=files)

    if response.status_code == 200:
        data = response.json()
        feedback = data["feedback"]
        processed_frames = data["processed_frames"]
        
        st.write("✅ AI Feedback on Skating Performance:")
        for i in range(min(10, len(feedback), len(processed_frames))):  # Show first 10 frames or less if fewer available
            frame_url = processed_frames[i]
            frame_name = frame_url.split("/")[-1]  # Extract filename
            img_response = requests.get(f"http://127.0.0.1:5001/get_frame/{frame_name}")
            
            if img_response.status_code == 200:
                img = Image.open(BytesIO(img_response.content))
                st.image(img, caption=f"Processed Frame {frame_name}")
            
            st.write(f"**Feedback for Frame {i+1}:**")
            feedback_list = feedback[i]
            # Feedback for Frame
            if "Feedback" in feedback_list and isinstance(feedback_list["Feedback"], list):
                st.write("**Feedback:**")
                for sub_item in feedback_list["Feedback"]:
                    if isinstance(sub_item, dict):
                        for sub_key, sub_value in sub_item.items():
                            st.write(f"- **{sub_key}**: {sub_value}")
                    else:
                        st.write(f"- {sub_item}")

            # Metrics for Frame
            st.write("**Metrics:**")
            for key, value in feedback_list.items():
                if key != "Feedback":
                    st.write(f"- **{key}**: {value}")
    else:
        st.write("❌ Error processing video.")
