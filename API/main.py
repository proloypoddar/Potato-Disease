from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os
import uuid
from gtts import gTTS
import tempfile
import shutil

app = FastAPI(title="Potato Disease Detector API",
              description="API for detecting potato plant diseases with TTS support",
              version="1.0.0")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define constants
MODEL_PATH = "E:/ALUuu/my_model_torch.pth"
IMG_SIZE = 224
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Suggestions in both English and Bengali
SUGGESTIONS = {
    "Early Blight": {
        "en": "Apply fungicide regularly and remove infected leaves to control Early Blight.",
        "bn": "আলু গাছে আরলি ব্লাইট প্রতিরোধে নিয়মিত ছত্রাকনাশক প্রয়োগ করুন এবং আক্রান্ত পাতাগুলি অপসারণ করুন।"
    },
    "Late Blight": {
        "en": "Ensure good drainage and quickly remove infected plant parts to prevent Late Blight spread.",
        "bn": "লেট ব্লাইট প্রতিরোধে ভাল নিষ্কাশন ব্যবস্থা নিশ্চিত করুন এবং সংক্রমিত গাছের অংশ দ্রুত কেটে ফেলুন।"
    },
    "Healthy": {
        "en": "Your plant is healthy! Continue regular care and preventive measures.",
        "bn": "আপনার গাছটি সুস্থ আছে! নিয়মিত পরিচর্যা এবং রোগ প্রতিরোধক ব্যবস্থা চালিয়ে যান।"
    }
}

# Create temp directory for audio files
TEMP_DIR = os.path.join(tempfile.gettempdir(), "potato_tts")
os.makedirs(TEMP_DIR, exist_ok=True)

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load the model
try:
    # Load model architecture
    model = models.mobilenet_v2(weights=None)
    num_classes = len(CLASS_NAMES)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Load saved weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # If model loading fails, we'll handle this in the predict endpoint

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/ping")
async def ping():
    return {"message": "Backend is alive!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check if model is loaded
        if 'model' not in globals():
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Read and process the image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Transform the image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

            # Get class and confidence
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

            return {
                "class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "suggestion": {
                    "en": SUGGESTIONS[predicted_class]["en"],
                    "bn": SUGGESTIONS[predicted_class]["bn"]
                }
            }

    except Exception as e:
        return {"error": str(e)}

@app.post("/text-to-speech")
async def text_to_speech(text: str = None, language: str = "en"):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")

    try:
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)

        # Generate TTS audio file
        tts = gTTS(text=text, lang=language)
        tts.save(filepath)

        # Return the audio file
        return FileResponse(
            path=filepath,
            media_type="audio/mpeg",
            filename=filename,
            background=None
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"TTS generation failed: {str(e)}"}
        )

@app.get("/copyright")
async def copyright():
    return {"developer": "Polok Poddar (Proloy)"}

# Use lifespan context manager for cleanup (modern approach)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing special to do
    yield
    # Shutdown: clean up temp files
    try:
        shutil.rmtree(TEMP_DIR)
        print("Cleaned up temporary TTS files")
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")

# Update app with lifespan
app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
