import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import torch
import numpy as np
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from torch.onnx import export

# --- 1. Загрузка модели ---
model_path = "patchcore_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(model_path, map_location=device)

# Создание модели с правильными параметрами
model = PatchcoreModel(
    layers=["layer2", "layer3"],
    backbone="wide_resnet50_2",
    pre_trained=True,
    num_neighbors=9
)

# Загрузка весов
state_dict = {}
for key, value in checkpoint.items():
    if key.startswith("model."):
        new_key = key[6:]
        state_dict[new_key] = value

model.load_state_dict(state_dict, strict=False)

# Восстановление memory_bank
if "model.memory_bank" in checkpoint:
    model.memory_bank = checkpoint["model.memory_bank"]

model.eval()
model.to(device)
print("✅ Модель успешно загружена")

# --- 2. Подготовка к конвертации ---

# Размер входного изображения (как в обучении)
input_size = (224, 224)
batch_size = 1

# Создаём пример входных данных
dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)

# --- 3. Конвертация в ONNX ---

# Путь для сохранения ONNX модели
onnx_path = "patchcore_model.onnx"

# Определяем, что возвращает модель, чтобы правильно задать output_names
print("🔍 Проверяем, что возвращает модель...")
with torch.no_grad():
    test_output = model(dummy_input)
    if isinstance(test_output, tuple):
        num_outputs = len(test_output)
        print(f"   Модель возвращает {num_outputs} тензоров")
        # Создаём имена для выходов
        output_names = [f"output_{i}" for i in range(num_outputs)]
        # Динамические оси для каждого выхода
        dynamic_axes = {'input': {0: 'batch_size'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}
    else:
        output_names = ["output"]
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    print(f"   Имена выходов: {output_names}")

# Экспорт в ONNX
print("⏳ Начинаем экспорт в ONNX...")
try:
    torch.onnx.export(
        model,                      # модель
        dummy_input,                # пример входных данных
        onnx_path,                  # путь для сохранения
        export_params=True,         # сохранять веса
        opset_version=18,           # версия ONNX (используем стабильную 18)
        do_constant_folding=True,   # оптимизация констант (безопасно)
        input_names=['input'],      # имена входов
        output_names=output_names,  # имена выходов (определены автоматически)
        dynamic_axes=dynamic_axes,  # динамические размеры
        verbose=False,              # отключить подробный вывод
        external_data=False         # сохранить все веса внутри .onnx (без .data)
    )
    print(f"✅ Модель успешно конвертирована в ONNX: {onnx_path}")
    
    # Проверка размера файла
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"   Размер файла: {size_mb:.2f} МБ")
        
        # Проверяем, не создался ли .data файл
        data_file = onnx_path + ".data"
        if os.path.exists(data_file):
            print(f"   ⚠️ Внимание: создан внешний файл {data_file} (размер {os.path.getsize(data_file) / (1024*1024):.2f} МБ)")
            print("   Это означает, что модель слишком большая для одного файла")
        else:
            print("   ✅ Все веса сохранены внутри .onnx (внешний .data файл не создан)")
    
except Exception as e:
    print(f"❌ Ошибка при конвертации: {e}")
    print("\n🔧 Пробуем альтернативный метод конвертации...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
            external_data=True  # разрешаем внешние данные, если модель слишком большая
        )
        print(f"✅ Модель успешно конвертирована с внешними данными: {onnx_path}")
        print("   📁 Созданы файлы: .onnx и .data (веса вынесены в .data)")
    except Exception as e2:
        print(f"❌ Ошибка при альтернативной конвертации: {e2}")

# --- 4. Проверка ONNX модели ---
print("\n🔍 Проверка ONNX модели...")

try:
    import onnx
    import onnxruntime as ort
    
    # Загрузка ONNX модели
    onnx_model = onnx.load(onnx_path)
    
    # Проверка валидности
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX модель валидна")
    
    # Информация о модели
    print("\n📊 Информация о ONNX модели:")
    print(f"  Версия opset: {onnx_model.opset_import[0].version}")
    print(f"  Количество входов: {len(onnx_model.graph.input)}")
    print(f"  Количество выходов: {len(onnx_model.graph.output)}")
    
    # Вывод входов и выходов
    print("\n  Входы:")
    for inp in onnx_model.graph.input:
        print(f"    - {inp.name}: {inp.type}")
    
    print("\n  Выходы:")
    for out in onnx_model.graph.output:
        print(f"    - {out.name}: {out.type}")
    
    # --- 5. Тестирование ONNX модели ---
    print("\n🧪 Тестирование ONNX модели...")
    
    # Создаём сессию ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Подготовка входных данных
    input_name = ort_session.get_inputs()[0].name
    test_input = np.random.randn(batch_size, 3, input_size[0], input_size[1]).astype(np.float32)
    
    # Запуск инференса
    ort_outputs = ort_session.run(None, {input_name: test_input})
    
    print(f"✅ Тест ONNX модели успешен")
    print(f"  Количество выходов: {len(ort_outputs)}")
    for i, out in enumerate(ort_outputs):
        print(f"  Выход {i} ({output_names[i] if i < len(output_names) else 'unknown'}): форма {out.shape}, тип {out.dtype}")
    
except ImportError as e:
    print(f"⚠️ onnx или onnxruntime не установлены: {e}")
    print("  Установите: pip install onnx onnxruntime")
except Exception as e:
    print(f"❌ Ошибка при проверке ONNX модели: {e}")

print("\n🎉 Конвертация завершена!")
