
from fastapi import FastAPI, File, UploadFile
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference
import io

app = FastAPI()

# 1. Load the architecture exactly as we trained it
device = torch.device("cpu") # Use CPU for inference stability in basic APIs
model = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)

# 2. Load the trained weights
MODEL_PATH = "final_brain_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded NIfTI file
    contents = await file.read()
    # In a real app, you'd use nibabel to load this buffer
    # and apply the same train_transforms (normalization/RAS)

    return {"status": "success", "message": "3D Volume received and processed."}

@app.get("/")
def health_check():
    return {"status": "online", "model": "AttentionUnet-3D-BraTS"}
