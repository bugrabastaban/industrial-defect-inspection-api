import os
import xml.etree.ElementTree as ET
import shutil


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]
class_to_id = {cls: idx for idx, cls in enumerate(CLASSES)}

def convert_coordinates(size, box):
    
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    
    x_center = round(x_center * dw, 6)
    y_center = round(y_center * dh, 6)
    w = round(w * dw, 6)
    h = round(h * dh, 6)
    return x_center, y_center, w, h

def process_split(split_name):
    print(f"[{split_name.upper()}] İşleniyor...")
    
    raw_images_dir = f"raw_dataset/{split_name}/images"
    raw_anno_dir = f"raw_dataset/{split_name}/annotations"
    
    dest_images_dir = f"yolov8_dataset/{split_name}/images"
    dest_labels_dir = f"yolov8_dataset/{split_name}/labels"
    
    
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    
    
    xml_files = [f for f in os.listdir(raw_anno_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(raw_anno_dir, xml_file)
        
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        
        size_elem = root.find('size')
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        
        yolo_lines = []
        
        
        for obj in root.findall('object'):
            cls_name = obj.find('name').text.strip()
            if cls_name not in class_to_id:
                continue
                
            class_id = class_to_id[cls_name]
            bndbox = obj.find('bndbox')
            
            xmin = float(bndbox.find('xmin').text)
            xmax = float(bndbox.find('xmax').text)
            ymin = float(bndbox.find('ymin').text)
            ymax = float(bndbox.find('ymax').text)
            
            
            x, y, w, h = convert_coordinates((width, height), (xmin, xmax, ymin, ymax))
            yolo_lines.append(f"{class_id} {x} {y} {w} {h}\n")
            
        
        if yolo_lines:
            
            txt_path = os.path.join(dest_labels_dir, f"{base_name}.txt")
            with open(txt_path, 'w') as f:
                f.writelines(yolo_lines)
                
            
            img_copied = False
            for root_path, _, files in os.walk(raw_images_dir):
                img_name = f"{base_name}.jpg"
                if img_name in files:
                    shutil.copy(os.path.join(root_path, img_name), os.path.join(dest_images_dir, img_name))
                    img_copied = True
                    break
            
            if not img_copied:
                print(f"⚠️ Uyarı: {base_name}.jpg görseli bulunamadı!")

    print(f"[{split_name.upper()}] Tamamlandı. Çıktılar 'yolov8_dataset' klasörüne aktarıldı.\n")

if __name__ == "__main__":
    process_split("train")
    process_split("validation")