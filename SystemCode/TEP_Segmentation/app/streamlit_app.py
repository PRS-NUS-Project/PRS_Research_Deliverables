import base64
import os
import shutil
import tempfile
import time

import dotenv
import streamlit as st
from moviepy import ImageSequenceClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow

dotenv.load_dotenv()
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.set_page_config(
    page_title="TEP Segmentation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Instructions")
    st.write(
        "Upload an image, video, or zip file containing frames for TEP segmentation."
    )
    st.write("- **Image**: Direct inference and display.")
    st.write("- **Video**: Extract frames or annotate entire video.")
    st.write("- **Zip**: Download extracted frames.")
    st.info("Supported formats: PNG, JPG, JPEG, MOV, MP4, AVI, ZIP")

st.title("TEP Segmentation Application 🧊")

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None
if "processed_video_bytes" not in st.session_state:
    st.session_state.processed_video_bytes = None
if "video_mode" not in st.session_state:
    st.session_state.video_mode = None


def initialize_model():
    try:
        model_predict = Roboflow(api_key="g9TuHO8utwgvXT0KEmnR")
        workspace = model_predict.workspace()
        if workspace is None:
            st.error("Failed to access workspace")
            return None
        project = workspace.project("tep-instance-segmentation-em54w")
        if project is None:
            st.error("Failed to access project")
            return None
        version = project.version(2)
        if version is None:
            st.error("Failed to access model version")
            return None
        model = version.model
        if model is None:
            st.error("Failed to load model")
            return None
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None


def add_labels_to_image(image_path, predictions, output_path, background=False):
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

    for pred in predictions.predictions:
        label = pred["class"]
        x = pred["x"]
        y = pred["y"]

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if background:
            bg_x = x - text_width // 2 - 5
            bg_y = y - text_height // 2 - 5
            bg_width = text_width + 10
            bg_height = text_height + 10
            draw.rectangle(
                [bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=(0, 123, 255, 200)
            )

        text_x = x - text_width // 2
        text_y = y - text_height // 2
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(img, overlay)
    result = result.convert("RGB")
    result.save(output_path)


col1, col2 = st.columns([3, 3])

with col1:
    st.subheader("Upload File")
    upload_file = st.file_uploader(
        "Upload a zip file containing image frames, a video, or an image",
        type=["zip", "mov", "mp4", "avi", "png", "jpg", "jpeg"],
    )

    if upload_file is not None and upload_file.type in [
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
    ]:
        if not st.session_state.video_processed:
            st.subheader("Video Annotation Options")
            video_option = st.radio(
                "Choose how to annotate the video:",
                ("Extract Frames (ZIP)", "Annotate Video (Run Model on All Frames)"),
                key="video_annotation_option",
            )

            if st.button("Annotate Video", key="annotate_button"):
                st.session_state.video_mode = (
                    "extract" if video_option == "Extract Frames (ZIP)" else "annotate"
                )

with col2:
    if upload_file is not None:
        st.subheader("Results")

        if upload_file.type in ["image/png", "image/jpg", "image/jpeg"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = os.path.join(temp_dir, "upload_file")
                with open(image_path, "wb") as f:
                    f.write(upload_file.getbuffer())
                    st.toast("Image uploaded successfully!")

                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                status_placeholder.write("Initializing model...")
                progress_placeholder.progress(25)

                model = initialize_model()
                if model is None:
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    st.stop()

                status_placeholder.write("Performing inference...")
                progress_placeholder.progress(50)

                predictions = model.predict(image_path)
                predictions.save(os.path.join(temp_dir, "predicted_image.png"))

                status_placeholder.write("Adding labels...")
                progress_placeholder.progress(75)

                annotated_path = os.path.join(temp_dir, "annotated_image.png")
                add_labels_to_image(
                    os.path.join(temp_dir, "predicted_image.png"),
                    predictions,
                    annotated_path,
                )

                progress_placeholder.progress(100)
                time.sleep(0.5)
                progress_placeholder.empty()
                status_placeholder.empty()

                with open(annotated_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()

                labels = [pred["class"] for pred in predictions.predictions]
                label_text = ", ".join(labels) if labels else "No detections"

                st.markdown(
                    f"""
                <div title="{label_text}">
                    <img src="data:image/png;base64,{img_base64}" width="350" style="border-radius: 10px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);">
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.toast("Inference completed successfully!")
        else:
            if st.session_state.video_mode and not st.session_state.video_processed:
                st.toast("Video uploaded successfully!")

                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                status_placeholder.write("Loading video...")

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_video:
                    temp_video.write(upload_file.getbuffer())
                    temp_video_path = temp_video.name
                clip = VideoFileClip(temp_video_path)

                with tempfile.TemporaryDirectory() as temp_dir:
                    frames_dir = os.path.join(temp_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)

                    total_frames = int(clip.fps * clip.duration)
                    progress_placeholder.progress(0)
                    status_placeholder.write(f"Extracting frames... (0/{total_frames})")

                    for i, frame in enumerate(clip.iter_frames()):
                        frame_image = Image.fromarray(frame)
                        frame_image.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

                        if i % max(1, total_frames // 20) == 0 or i == total_frames - 1:
                            progress_value = min((i + 1) / total_frames, 1.0)
                            progress_placeholder.progress(progress_value)
                            status_placeholder.write(
                                f"Extracting frames... ({i+1}/{total_frames})"
                            )
                            time.sleep(0.01)

                    if st.session_state.video_mode == "extract":
                        status_placeholder.write("Creating zip file...")
                        progress_placeholder.progress(1.0)
                        time.sleep(0.1)

                        zip_path = shutil.make_archive(
                            os.path.join(temp_dir, "extracted_frames"),
                            "zip",
                            frames_dir,
                        )
                        with open(zip_path, "rb") as f:
                            st.session_state.zip_bytes = f.read()

                        progress_placeholder.empty()
                        status_placeholder.empty()
                    else:
                        annotated_frames_dir = os.path.join(
                            temp_dir, "annotated_frames"
                        )
                        os.makedirs(annotated_frames_dir, exist_ok=True)

                        status_placeholder.write("Initializing model...")
                        progress_placeholder.progress(0)
                        time.sleep(0.1)

                        model = initialize_model()
                        if model is None:
                            progress_placeholder.empty()
                            status_placeholder.empty()
                            clip.close()
                            os.unlink(temp_video_path)
                            st.stop()

                        frame_files = sorted(
                            [f for f in os.listdir(frames_dir) if f.endswith(".png")]
                        )
                        total_frames = len(frame_files)

                        status_placeholder.write(
                            f"Annotating frames with model... (0/{total_frames})"
                        )

                        for idx, frame_file in enumerate(frame_files):
                            frame_path = os.path.join(frames_dir, frame_file)
                            predictions = model.predict(frame_path)
                            temp_pred_path = os.path.join(temp_dir, "temp_pred.png")
                            predictions.save(temp_pred_path)

                            output_path = os.path.join(annotated_frames_dir, frame_file)
                            add_labels_to_image(
                                temp_pred_path, predictions, output_path
                            )

                            if (
                                idx % max(1, total_frames // 20) == 0
                                or idx == total_frames - 1
                            ):
                                progress_value = min((idx + 1) / total_frames, 1.0)
                                progress_placeholder.progress(progress_value)
                                status_placeholder.write(
                                    f"Annotating frames with model... ({idx+1}/{total_frames})"
                                )
                                time.sleep(0.01)

                        status_placeholder.write("Creating annotated video...")
                        progress_placeholder.progress(1.0)
                        time.sleep(0.1)

                        annotated_frame_files = sorted(
                            [
                                os.path.join(annotated_frames_dir, f)
                                for f in os.listdir(annotated_frames_dir)
                                if f.endswith(".png")
                            ]
                        )
                        annotated_clip = ImageSequenceClip(
                            annotated_frame_files, fps=clip.fps
                        )

                        output_video_path = os.path.join(
                            temp_dir, "annotated_video.mp4"
                        )
                        annotated_clip.write_videofile(
                            output_video_path, codec="libx264", audio=False, logger=None
                        )

                        with open(output_video_path, "rb") as f:
                            st.session_state.processed_video_bytes = f.read()

                        annotated_clip.close()
                        progress_placeholder.empty()
                        status_placeholder.empty()

                clip.close()
                os.unlink(temp_video_path)
                st.session_state.video_processed = True

            if st.session_state.video_processed:
                if (
                    st.session_state.video_mode == "extract"
                    and st.session_state.zip_bytes
                ):
                    download_button = st.download_button(
                        label="Download Extracted Frames as ZIP",
                        data=st.session_state.zip_bytes,
                        file_name="extracted_frames.zip",
                        mime="application/zip",
                        key="download_zip",
                    )
                    if download_button:
                        st.toast("Downloaded successfully!")
                elif (
                    st.session_state.video_mode == "annotate"
                    and st.session_state.processed_video_bytes
                ):
                    download_button = st.download_button(
                        label="Download Annotated Video",
                        data=st.session_state.processed_video_bytes,
                        file_name="annotated_video.mp4",
                        mime="video/mp4",
                        key="download_video",
                    )
                    if download_button:
                        st.toast("Downloaded successfully!")
