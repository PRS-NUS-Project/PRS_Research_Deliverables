# Development and Evaluation of an AI Model for Identifying Critical Anatomical Structures in Surgical Operation Video

## 📋 Project Description

This project focuses on the development of an AI-powered segmentation model for identifying critical anatomical structures in surgical operation videos, specifically targeting Transabdominal Preperitoneal(TEP) procedures. The primary deliverable is a user-friendly Streamlit web application that allows medical professionals and researchers to upload images, videos, or frame sequences for real-time inference using a pre-trained Roboflow model. The application supports direct image segmentation, video frame extraction, and full video annotation, providing labeled outputs to assist in surgical analysis and training.

Key objectives include:

- Accurate segmentation of anatomical structures in endoscopic videos.
- Easy-to-use interface for non-technical users.
- Docker-based deployment for portability and scalability.
- Integration with Roboflow for model management and inference.

## ✨ Features

- **Image Segmentation**: Upload individual images (PNG, JPG, JPEG) for instant segmentation and labeling.
- **Video Processing**: Upload videos (MOV, MP4, AVI) with options to extract frames as a ZIP file or annotate the entire video with segmentation overlays.
- **Batch Processing**: Handle ZIP files containing multiple image frames for bulk inference.
- **Model Integration**: Utilizes a YOLOv11 instance segmentation modelfor predictions.
- **User-Friendly UI**: Streamlit-based interface with progress tracking, download options, and responsive design.

## 👥 Contributors

- Arshi Saxena
- Norbert Oliver
- Pranjali Sonawane
- Sharvesh Subhash

## 🛠️ Project Setup

## 1. Production Deployment (for end users)

To run the application, ensure Docker Desktop is installed and running on your system.

You can download it from the official site: https://www.docker.com/products/docker-desktop

### Run the Docker image

`docker load -i tep_segmentation_image.tar`

`docker run -p 8501:8501 tep_segmentation`

Open http://localhost:8501 in your browser.

## 2. Development Environment

This project uses **Python 3.12** and requires a virtual environment to manage dependencies. Follow these steps to set up the project locally:

### 1. Clone the repository

`git clone https://github.com/PRS-NUS-Project/PRS_Research_Deliverables.git`

### 2. Python Version

Ensure you have python version **Python 3.12.x** installed before proceeding with the next steps.

### 3. Run the setup script

`cd <your_project_path>/TEP_Segmentation`

#### Windows

`.\setup_env.bat`

#### macOS/Linux

`bash setup_env.sh`

### 4. Activate the virtual environment

This project uses **VS Code auto-activation** as configured in `.vscode/settings.json`.

- **If you are using VS Code:**  
  Ensure that Python extension is installed and enabled. Opening a new terminal(cmd) in VS Code for this project will automatically activate the virtual environment, and it will automatically deactivate when you close VS Code.
  **Check:** (.venv) should be appended to the path in cmd terminal in VS Code for successful auto-activation
  `(.venv) path\TEP_Segmentation>`

- **If you are using any other IDE or terminal:**  
  You will need to manually activate the environment **each time** you open the project:

  #### Windows - Powershell

  `.venv\Scripts\activate.bat`

  #### macOS/Linux

  `source .venv/bin/activate`

### 5. Model Training

From Project Root: `python -m src.main`

### 6. Start the Frontend (Streamlit)

`streamlit run app/streamlit_app.py`

- The frontend will run at: http://localhost:8501

### 7. Build and Export the Docker Image (for release)

Ensure Docker Desktop is installed and running on your system.

You can download it from the official site:
👉 https://www.docker.com/products/docker-desktop

#### Build the Docker image

`docker buildx build --load -t tep_segmentation .`

#### Export the image as a .tar file for distribution

`docker save -o tep_segmentation_image.tar tep_segmentation`
