---
title: Industrial Defect Inspection API
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Endüstriyel Hata Tespiti ve MLOps API

Bu proje, üretim hatlarındaki yüzey kusurlarını (çizik, yama, vb.) gerçek zamanlı olarak tespit eden ve hatalı üretimleri bulut ortamına (S3 simülasyonu) loglayan uçtan uca bir makine öğrenmesi boru hattıdır (MLOps pipeline).

## 🚀 Teknolojiler
* **Yapay Zeka:** Ultralytics YOLOv8 (Custom Trained on NEU-DET)
* **Backend:** FastAPI, Uvicorn
* **Görüntü İşleme:** OpenCV, Pillow
* **Dağıtım (Deployment):** Docker

## 🔌 API Kullanımı (Postman)
Sistemi test etmek için `/predict` uç noktasına bir `POST` isteği gönderebilirsiniz.

* **Endpoint:** `POST /predict`
* **Body Formatı:** `multipart/form-data`
* **Key:** `file` (Görsel dosyası yüklenmelidir)

API, görseli işleyerek tespit edilen hataların koordinatlarını, güven skorlarını ve MLOps loglama durumunu JSON formatında döndürecektir.