#!/usr/bin/env python3
"""
Минимальный скрипт для инференса ONNX модели через OpenVINO
"""

import cv2
import numpy as np
import argparse
from openvino import Core

# Конфиг классов
CLASSES = ["LightVehicle", "Person", "Building", "UPole", "Boat", "Bike", 
           "Container", "Truck", "Gastank", "Digger", "SolarPanels", "Bus"]

def parse_args():
    parser = argparse.ArgumentParser(description='OpenVINO ONNX Inference')
    parser.add_argument('model', help='Путь к ONNX модели')
    parser.add_argument('image', help='Путь к изображению')
    parser.add_argument('-c', '--conf', type=float, default=0.4, 
                        help='Порог уверенности (по умолчанию: 0.4)')
    parser.add_argument('-o', '--output', help='Путь для сохранения результата')
    parser.add_argument('-d', '--device', default='CPU', 
                        help='Устройство: CPU, GPU, AUTO (по умолчанию: CPU)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"📦 Загрузка модели: {args.model}")
    print(f"🖼️  Изображение: {args.image}")
    print(f"🎯 Порог: {args.conf}")
    print(f"💻 Устройство: {args.device}")
    
    # Загрузка модели
    core = Core()
    model = core.read_model(args.model)
    compiled_model = core.compile_model(model, args.device)
    input_size = compiled_model.input(0).shape[2]
    
    # Загрузка изображения
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {args.image}")
    
    h, w = image.shape[:2]
    
    # Обрезка до квадрата
    size = min(h, w)
    start_y, start_x = (h - size)//2, (w - size)//2
    cropped = image[start_y:start_y+size, start_x:start_x+size]
    resized = cv2.resize(cropped, (input_size, input_size))
    
    # Тензор
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    
    # Инференс
    result = compiled_model([tensor])[compiled_model.output(0)]
    predictions = np.transpose(result, (0, 2, 1))[0]
    
    # Постобработка
    detections = []
    for pred in predictions:
        cx, cy, wb, hb = pred[:4]
        scores = pred[4:]
        class_id = np.argmax(scores)
        conf = scores[class_id]
        
        if conf > 1.0:
            conf = conf / 255.0
        
        if conf > args.conf:
            scale = size / input_size
            x1 = int((cx - wb/2) * input_size * scale + start_x)
            y1 = int((cy - hb/2) * input_size * scale + start_y)
            x2 = int((cx + wb/2) * input_size * scale + start_x)
            y2 = int((cy + hb/2) * input_size * scale + start_y)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, conf, class_id))
    
    # Вывод
    print(f"\nНайдено объектов: {len(detections)}")
    for x1, y1, x2, y2, conf, cls in detections:
        class_name = CLASSES[cls] if cls < len(CLASSES) else f"Class_{cls}"
        print(f"  {class_name}: {conf:.3f} -> [{x1},{y1},{x2},{y2}]")
    
    # Сохранение
    output_path = args.output or args.image.replace('.', '_openvino.')
    result_img = image.copy()
    
    for x1, y1, x2, y2, conf, cls in detections:
        class_name = CLASSES[cls] if cls < len(CLASSES) else f"Class_{cls}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_img, f"{class_name}:{conf:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, result_img)
    print(f"\n✅ Сохранено: {output_path}")

if __name__ == "__main__":
    main()
