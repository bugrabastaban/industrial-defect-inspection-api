from ultralytics import YOLO

def main():
    
    model = YOLO("yolov8n.pt")

    
    print(" Model eğitimi başlatılıyor...")
    results = model.train(
        data="data.yaml",             
        epochs=50,                    
        imgsz=416,                    
        batch=8,                      
        device=0,                     
        name="industrial_defect_v1",  
        plots=True                    
    )
    print(" Eğitim tamamlandı!")

if __name__ == '__main__':
    main()