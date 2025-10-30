import streamlit as st
import os
import tempfile
import shutil
import zipfile
import time
import base64
from roboflow import Roboflow
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy import VideoFileClip
from PIL import Image



st.markdown(
  '''
<style>
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f0f8ff;
    color: #333;
  }
  h1 {
    text-align: center;
    color: #007bff;
    font-weight: bold;
    margin-bottom: 20px;
    margin-top:-60px !important;
  }
  .stFileUploader {
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 123, 255, 0.2);
    background-color: black;
    font-weight: bold;
    transition: all 0.3s ease;
  }
  .stFileUploader:hover {
    box-shadow: 0 6px 30px rgba(0, 123, 255, 0.3);
    border-color: #0056b3;
  }
  .stButton > button {
    background: linear-gradient(45deg, #007bff, #0056b3);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 10px 20px;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3);
  }
  .stButton > button:hover {
    background: linear-gradient(45deg, #0056b3, #004085);
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0, 123, 255, 0.4);
  }
  .stImage > img {
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    max-width: 100%;
    height: auto;
  }
  .sidebar .sidebar-content {
    background-color: #e3f2fd;
    padding: 20px;
    border-radius: 10px;
  }
  .stProgress > div > div > div {
    background-color: #007bff;
  }
  .stSuccess, .stInfo, .stError {
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
  }
  @media (max-width: 768px) {
    .stFileUploader {
      padding: 10px;
    }
    .stButton > button {
      width: 100%;
    }
  }
  labels{
  text-align:center !important
  }
  img:hover{
  cursor:pointer;
  }
  

</style>
'''
, unsafe_allow_html=True)

# page configuration

st.set_page_config(
  page_title= 'TEP Segmentation',
  page_icon='🧊',
  layout= 'wide',
  initial_sidebar_state='expanded',
)

# sidebar for instructions
with st.sidebar:
    st.header("Instructions")
    st.write("Upload an image, video, or zip file containing frames for TEP segmentation.")
    st.write("- **Image**: Direct inference and display.")
    st.write("- **Video**: Extract frames from video.")
    st.write("- **Zip**: Download extracted frames.")
    st.info("Supported formats: PNG, JPG, JPEG, MOV, MP4, AVI, ZIP")

# title of the app
st.title("TEP Segmentation Application 🧊")

# Initialize session state for video processing
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'zip_bytes' not in st.session_state:
    st.session_state.zip_bytes = None

# main content with columns
col1, col2 = st.columns([3, 3])

with col1:
    st.subheader("Upload File")
    upload_file = st.file_uploader("Upload a zip file containing image frames, a video, or an image", type=["zip", "mov", "mp4", "avi", "png", "jpg", "jpeg"])

with col2:
    if upload_file is not None:
        st.subheader("Results")

        if upload_file.type in ['image/png', 'image/jpg', 'image/jpeg']:
            # saving the uploaded image files to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path=os.path.join(temp_dir, 'upload_file')
                with open(image_path,'wb') as f:
                    f.write(upload_file.getbuffer())
                    st.toast("Image uploaded successfully!")

                # using model for inference
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                progress_placeholder.progress(0)
                status_placeholder.write("Initializing model...")
                progress_placeholder.progress(25)
                time.sleep(1)
                model_input=image_path
                model_predict=Roboflow(api_key='g9TuHO8utwgvXT0KEmnR')
                status_placeholder.write("Loading workspace...")
                progress_placeholder.progress(50)
                time.sleep(1)
                project=model_predict.workspace().project('tep-instance-segmentation-em54w')
                model=project.version(2).model
                status_placeholder.write("Performing inference...")
                progress_placeholder.progress(75)
                time.sleep(1)
                predictions = model.predict(model_input)
                predictions.save(os.path.join(temp_dir, 'predicted_image.png'))
                progress_placeholder.progress(100)
                time.sleep(1)
                progress_placeholder.empty()
                status_placeholder.empty()
                # Encode image to base64 for tooltip
                with open(os.path.join(temp_dir, 'predicted_image.png'), 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                # Extract and display labels
                labels = [pred['class'] for pred in predictions.predictions]
                label_text = ", ".join(labels)
                # Display image with tooltip
                st.markdown(f"""
                <div title="{label_text}">
                    <img src="data:image/png;base64,{img_base64}" width="350" style="border-radius: 10px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);">
                </div>
                """, unsafe_allow_html=True)
                st.toast("Inference completed successfully!")
        else:
            if not st.session_state.video_processed:
                st.toast("Video uploaded successfully!")

                # extracting frames from the video
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                status_placeholder.write("Loading video...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                    temp_video.write(upload_file.getbuffer())
                    temp_video_path = temp_video.name
                clip=VideoFileClip(temp_video_path)
                with tempfile.TemporaryDirectory() as temp_dir:
                    frames_dir=os.path.join(temp_dir, 'frames')
                    os.makedirs(frames_dir, exist_ok=True)

                    total_frames = int(clip.fps * clip.duration)
                    progress_placeholder.progress(0)
                    status_placeholder.write("Extracting frames...")
                    with st.spinner('Extracting frames from video...'):
                        for i, frame in enumerate(clip.iter_frames()):
                            frame_image=Image.fromarray(frame)
                            frame_image.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
                            progress_placeholder.progress(min((i + 1) / total_frames, 1.0))
                    progress_placeholder.empty()
                    status_placeholder.write("Creating zip file...")
                    # Create zip after extraction
                    zip_path = shutil.make_archive(os.path.join(temp_dir, 'extracted_frames'), 'zip', frames_dir)
                    with open(zip_path, 'rb') as f:
                        st.session_state.zip_bytes = f.read()
                clip.close()  # Close the video clip to release the file
                os.unlink(temp_video_path)  # Clean up the temporary video file
                status_placeholder.empty()
                st.session_state.video_processed = True

            if st.session_state.video_processed and st.session_state.zip_bytes:
                download_button = st.download_button(
                    label="Download Extracted Frames as ZIP",
                    data=st.session_state.zip_bytes,
                    file_name="extracted_frames.zip",
                    mime="application/zip",
                    key="download_zip"
                )
                if download_button:
                    st.toast("Downloaded successfully!")
        



