from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import io
import cv2
import os


LOG_DIR = "s3_sim_bucket/defects_log"
os.makedirs(LOG_DIR, exist_ok=True)


app = FastAPI(
    title="Endüstriyel Hata Tespiti & MLOps API", 
    description="Hataları tespit eden ve kusurlu üretimleri loglama sistemi.",
    version="2.0"
)


MODEL_PATH = "endustriyel_model.pt"
try:
    model = YOLO(MODEL_PATH)
    print("Model ve MLOps modülü başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Model yüklenirken hata: {e}")

@app.get("/")
def root():
    return {"durum": "aktif", "sistem": "Hazır"}


@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    try:
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        
        results = model.predict(image, conf=0.25)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "hata_turu": model.names[int(box.cls[0].item())],
                    "guven_skoru": round(box.conf[0].item(), 2),
                    "koordinatlar": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)}
                })
        
        
        if len(detections) > 0:
            
            annotated_img = results[0].plot() 
            
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"defect_{timestamp}_{file.filename}"
            save_path = os.path.join(LOG_DIR, saved_filename)
            
            
            cv2.imwrite(save_path, annotated_img)
            
            log_mesaji = f"Görsel {LOG_DIR} klasörüne loglandı."
        else:
            log_mesaji = "Ürün kusursuz, loglama yapılmadı."

        return JSONResponse(content={
            "dosya_adi": file.filename,
            "tespit_edilen_hata_sayisi": len(detections),
            "mlops_durumu": log_mesaji,
            "hatalar": detections
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"hata": str(e)})