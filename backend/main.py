from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import base64
import numpy as np
import cv2
import uuid
from io import BytesIO
from pydantic import BaseModel
from typing import Optional
import time

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
STATIC_DIR = "backend/static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ProcessResponse(BaseModel):
    image_url: str
    score: Optional[float] = None
    time_ms: float
    recognition_time_ms: float

@app.get("/")
def read_root():
    return {"message": "AI Backend is running"}

from backend.ai_service import AIService
ai_service = AIService()

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        cleanup_static() # Clean up before saving new image
        session_id = str(uuid.uuid4())
        file_location = os.path.join(STATIC_DIR, f"original_{session_id}.jpg")
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        # Verify it's an image
        img = cv2.imread(file_location)
        if img is None:
             os.remove(file_location)
             raise HTTPException(status_code=400, detail="Invalid image file")
             
        return {"url": f"/static/original_{session_id}.jpg", "status": "success", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/glasses")
async def process_glasses(session_id: str):
    try:
        input_path = os.path.join(STATIC_DIR, f"original_{session_id}.jpg")
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="Original image not found")
            
        result_img, process_time = ai_service.add_sunglasses(input_path)
        output_filename = f"glasses_{session_id}.jpg"
        output_path = os.path.join(STATIC_DIR, output_filename)
        cv2.imwrite(output_path, result_img)
        
        score, rec_time = ai_service.recognize_face(output_path, input_path)
        
        return {
            "image_url": f"/static/{output_filename}?t={time.time()}",
            "score": score,
            "time_ms": process_time,
            "recognition_time_ms": rec_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/remove-glasses")
async def process_remove_glasses(session_id: str):
    try:
        input_path = os.path.join(STATIC_DIR, f"glasses_{session_id}.jpg")
        if not os.path.exists(input_path):
             raise HTTPException(status_code=404, detail="Image with glasses not found")
        
        original_path = os.path.join(STATIC_DIR, f"original_{session_id}.jpg")

        result_img, process_time = ai_service.process_reconstruction(input_path, original_path, 3)
        output_filename = f"inpainted_{session_id}.jpg"
        output_path = os.path.join(STATIC_DIR, output_filename)
        cv2.imwrite(output_path, result_img)
        
        score, rec_time = ai_service.recognize_face(output_path, original_path)
        
        return {
            "image_url": f"/static/{output_filename}?t={time.time()}",
            "score": score,
            "time_ms": process_time,
            "recognition_time_ms": rec_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/method/{method_id}")
async def process_method(method_id: int, session_id: str):
    try:
        input_path = os.path.join(STATIC_DIR, f"glasses_{session_id}.jpg")
        if not os.path.exists(input_path):
             raise HTTPException(status_code=404, detail="Image with glasses not found")
        original_path = os.path.join(STATIC_DIR, f"original_{session_id}.jpg")
        
        result_img, process_time = ai_service.process_reconstruction(input_path, original_path, method_id)
        
        output_filename = f"method_{method_id}_{session_id}.jpg"
        output_path = os.path.join(STATIC_DIR, output_filename)
        cv2.imwrite(output_path, result_img)
        
        score, rec_time = ai_service.recognize_face(output_path, original_path)
        
        return {
            "image_url": f"/static/{output_filename}?t={time.time()}",
            "score": score,
            "time_ms": process_time,
            "recognition_time_ms": rec_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
def reset():
    # In batch mode, we might want to keep files until session end, or delete everything
    # For now, let's keep it simple
    return {"status": "reset"}

def cleanup_static():
    """Keep only the latest files to avoid disk space issues during batch processing."""
    try:
        files = []
        for f in os.listdir(STATIC_DIR):
            full_path = os.path.join(STATIC_DIR, f)
            if os.path.isfile(full_path) and not f.startswith('.'):
                files.append(full_path)
        
        # Keep last 100 files for batch processing (allows ~16-17 images in progress)
        max_files = 100
        
        if len(files) > max_files:
            # Sort by modification time (oldest first)
            files.sort(key=os.path.getmtime)
            # Remove oldest files to keep only max_files
            files_to_remove = files[:-max_files]
            removed_count = 0
            for f in files_to_remove:
                try:
                    os.remove(f)
                    removed_count += 1
                except Exception as e:
                    print(f"Failed to remove {f}: {e}")
            
            if removed_count > 0:
                print(f"Cleanup: removed {removed_count} old files, {len(files) - removed_count} files remaining")
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        limit_max_requests=None,  # No limit on number of requests (for batch processing)
        timeout_keep_alive=300,   # 5 minutes keep-alive
        limit_concurrency=None,   # No concurrency limit
        timeout_graceful_shutdown=30
    )
