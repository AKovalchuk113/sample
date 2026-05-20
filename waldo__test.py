# Максимально простой скрипт с OpenVINO

import cv2
import numpy as np
from openvino import Core
import os
os.chdir('/Users/aleksandrkovalcuk/Downloads/')

CLASSES = ["LightVehicle", "Person", "Building", "UPole", "Boat", "Bike", 
           "Container", "Truck", "Gastank", "Digger", "SolarPanels", "Bus"]

# Загрузка модели
core = Core()
model = core.read_model("model.onnx")
compiled_model = core.compile_model(model, "CPU")
input_size = compiled_model.input(0).shape[2]

# Загрузка изображения
image = cv2.imread("road.png")
h, w = image.shape[:2]

# Обрезка до квадрата
size = min(h, w)
start_y, start_x = (h - size)//2, (w - size)//2
cropped = image[start_y:start_y+size, start_x:start_x+size]
resized = cv2.resize(cropped, (input_size, input_size))

# Тензор
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
tensor = np.expand_dims(np.transpose(rgb.astype(np.float32)/255.0, (2,0,1)), 0)

# Инференс
result = compiled_model([tensor])[compiled_model.output(0)]
predictions = np.transpose(result, (0,2,1))[0]

# Постобработка
detections = []
for pred in predictions:
    cx, cy, wb, hb = pred[:4]
    scores = pred[4:]
    class_id = np.argmax(scores)
    conf = scores[class_id]
    
    if conf > 0.4:
        scale = size / input_size
        x1 = int((cx - wb/2) * input_size * scale + start_x)
        y1 = int((cy - hb/2) * input_size * scale + start_y)
        x2 = int((cx + wb/2) * input_size * scale + start_x)
        y2 = int((cy + hb/2) * input_size * scale + start_y)
        detections.append((x1, y1, x2, y2, conf, class_id))

# Вывод
print(f"Найдено: {len(detections)} объектов")
for x1, y1, x2, y2, conf, cls in detections:
    print(f"  {CLASSES[cls]}: {conf:.3f} -> [{x1},{y1},{x2},{y2}]")

# Сохранение
result = image.copy()
for x1, y1, x2, y2, conf, cls in detections:
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(result, f"{CLASSES[cls]}:{conf:.2f}", (x1, y1-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("result_openvino.jpg", result)
print("✅ Сохранено в result_openvino.jpg")
