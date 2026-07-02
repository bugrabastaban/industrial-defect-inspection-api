import torch

# --- PYTORCH 2.6+ GÜVENLİK DUVARINI AŞMA (MONKEY PATCH) ---
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
# ---------------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import io
import cv2
import os

# --- SQLALCHEMY VERİTABANI AYARLARI ---
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./defects.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Veritabanı Tablo Modeli (ORM)
class DefectLog(Base):
    __tablename__ = "defect_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    dosya_adi = Column(String, index=True)
    hata_turu = Column(String)
    guven_skoru = Column(Float)
    koordinatlar = Column(String)  # JSON string olarak saklayacağız
    gorsel_yolu = Column(String)
    kayit_tarihi = Column(DateTime, default=datetime.utcnow)

# Tabloları otomatik oluştur (Eğer yoksa)
Base.metadata.create_all(bind=engine)
# ---------------------------------------

# MLOps Görsel Deposu Klasörü
LOG_DIR = "s3_sim_bucket/defects_log"
os.makedirs(LOG_DIR, exist_ok=True)

# API Uygulamasını Başlat
app = FastAPI(
    title="Endüstriyel Hata Tespiti & Veritabanı API", 
    description="Hataları tespit eden, SQL veritabanına kaydeden ve görselleri loglayan gelişmiş sistem.",
    version="3.0"
)

# Modeli Yükle
MODEL_PATH = "endustriyel_model.pt"
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model ve Veritabanı başarıyla senkronize edildi!")
except Exception as e:
    print(f"❌ Model yüklenirken hata: {e}")

@app.get("/")
def root():
    return {"durum": "aktif", "sistem": "Veritabanı Entegrasyonu Hazır"}

# 🚀 1. Endpoint: Görsel Analiz ve Veritabanına Kayıt
@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    db = SessionLocal() # Veritabanı oturumunu aç
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
        
        # Eğer hata tespit edildiyse hem görsele hem DB'ye logla
        if len(detections) > 0:
            annotated_img = results[0].plot() 
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"defect_{timestamp}_{file.filename}"
            save_path = os.path.join(LOG_DIR, saved_filename)
            cv2.imwrite(save_path, annotated_img)
            
            # --- VERİTABANINA YAZMA İŞLEMİ ---
            for d in detections:
                db_log = DefectLog(
                    dosya_adi=file.filename,
                    hata_turu=d["hata_turu"],
                    guven_skoru=d["guven_skoru"],
                    koordinatlar=str(d["koordinatlar"]), # Sözlüğü string'e çevirip kaydediyoruz
                    gorsel_yolu=save_path
                )
                db.add(db_log)
            db.commit() # Değişiklikleri kaydet
            # ----------------------------------
            
            log_mesaji = f"Görsel diske ve meta veriler SQLite veritabanına kaydedildi."
        else:
            log_mesaji = "Ürün kusursuz, kayıt yapılmadı."

        return JSONResponse(content={
            "dosya_adi": file.filename,
            "tespit_edilen_hata_sayisi": len(detections),
            "sistem_durumu": log_mesaji,
            "hatalar": detections
        })
        
    except Exception as e:
        db.rollback() # Hata durumunda işlemi geri al
        return JSONResponse(status_code=500, content={"hata": str(e)})
    finally:
        db.close() # Oturumu her halükarda kapat

# 🚀 2. Endpoint: Veritabanı Loglarını Listeleme (Yeni!)
@app.get("/logs")
def get_logs():
    db = SessionLocal()
    try:
        # Son eklenen hataları en üstte getirecek şekilde sorgula
        logs = db.query(DefectLog).order_by(DefectLog.id.desc()).all()
        
        result = []
        for log in logs:
            result.append({
                "id": log.id,
                "dosya_adi": log.dosya_adi,
                "hata_turu": log.hata_turu,
                "guven_skoru": log.guven_skoru,
                "koordinatlar": log.koordinatlar,
                "gorsel_yolu": log.gorsel_yolu,
                "kayit_tarihi": log.kayit_tarihi.strftime("%Y-%m-%d %H:%M:%S")
            })
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"hata": f"Veritabanından okunamadı: {str(e)}"})
    finally:
        db.close()