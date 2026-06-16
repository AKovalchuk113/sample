import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from openvino import Core
import matplotlib.pyplot as plt
import time

# --- 1. Инференс через OpenVINO ---
start_time = time.time()

model_path = "patchcore_model.onnx"
input_size = (224, 224)

core = Core()
compiled_model = core.compile_model(model_path, "CPU")

image_path = "009.png"

# Предобработка
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_full = Image.open(image_path).convert("RGB")
image_full_np = np.array(image_full)
image_tensor = transform(image_full).unsqueeze(0).numpy()

# Инференс
results = compiled_model([image_tensor])

# Извлечение результатов (адаптивное)
outputs = [results[out] for out in results]

# Ищем карту аномалий и оценку
pred_score = 0.0
anomaly_map = None

for out in outputs:
    if out.size == 1:
        pred_score = float(out)
        continue
    if out.size > 1 and len(out.shape) >= 2:
        temp = out.squeeze()
        if len(temp.shape) == 3 and temp.shape[0] == 1:
            temp = temp[0]
        if len(temp.shape) == 2:
            anomaly_map = temp
            break

if anomaly_map is None:
    print("❌ Карта аномалий не найдена!")
    exit(1)

print(f"Уровень аномальности: {pred_score:.4f}")
print(f"Размер карты аномалий: {anomaly_map.shape}")

# --- 2. Визуализация с alpha=0.25 ---
# Масштабирование
h_orig, w_orig = image_full_np.shape[:2]
anomaly_resized = cv2.resize(
    anomaly_map.astype(np.float32),
    (w_orig, h_orig),
    interpolation=cv2.INTER_LINEAR
)

# Нормализация
anomaly_norm = (anomaly_resized - anomaly_resized.min()) / (anomaly_resized.max() - anomaly_resized.min() + 1e-8)

# Тепловая карта
heatmap = plt.cm.jet(anomaly_norm)[:, :, :3]
heatmap = (heatmap * 255).astype(np.uint8)

# Наложение с alpha=0.25
alpha = 0.16
blended = (image_full_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)

# Сохранение
cv2.imwrite("overlay_alpha_025.png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
print(f"✅ Наложение сохранено как overlay_alpha_025.png")

# --- 3. Дополнительно: бинаризация ---
pixel_thr = np.percentile(anomaly_resized, 95)
binary_mask = (anomaly_resized > pixel_thr).astype(np.uint8) * 255
cv2.imwrite("binary_mask.png", binary_mask)
print(f"✅ Бинарная маска сохранена (порог: {pixel_thr:.4f})")

# --- 4. Статистика ---
print(f"\n📊 Статистика:")
print(f"  Уровень аномальности: {pred_score:.4f}")
print(f"  {'🔴 ДЕФЕКТ' if pred_score > 0.5 else '🟢 НОРМАЛЬНО'}")
print(f"  Дефектных пикселей: {np.sum(binary_mask > 0)}")
print(f"  Процент: {100 * np.sum(binary_mask > 0) / binary_mask.size:.2f}%")
print(f"  Время: {(time.time() - start_time)*1000:.2f} мс")

print("\n✅ Готово!")
