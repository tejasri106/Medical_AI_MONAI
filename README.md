# 3D Brain Tumor Segmentation

This project implements a Deep Learning pipeline to segment brain tumors from multi-modal MRI scans. It covers the entire lifecycle: from raw medical data preprocessing to deploying a containerized API.

## Overview
* **Dataset:** Medical Segmentation Decathlon (BraTS Task) 3D MRI scans.
* **Model:** 3D Attention U-Net (built with PyTorch and MONAI).
* **Deployment:** Production-ready FastAPI service containerized with Docker.

## Tech Stack
* **AI Frameworks:** PyTorch, MONAI
* **Medical Imaging:** Nibabel (NIfTI processing), NumPy
* **Backend:** FastAPI, Uvicorn
* **DevOps:** Docker


## Project Structure
* `3D_Brain_Segmentation.ipynb`: Data preprocessing, model training, and evaluation logic.
* `main.py`: The FastAPI script for serving the model via a web API.
* `Dockerfile`: Configuration to package the application into a portable container.
* `requirements.txt`: Python dependencies list.


## How to Use
> **Note:** Due to storage limits, the raw dataset and trained model weights (.pth) are not included in this repository.

1. **Download Data:** Get the BraTS/MSD dataset and place it in the `MedicalData/` folder.
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the API:**
   ```bash
   uvicorn main:app --reload
4. **Docker Deployment:**
   ```bash
   docker build -t brain-segmentation-api .
   docker run -p 8000:8000 brain-segmentation-api
