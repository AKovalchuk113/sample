import os
import torch
import numpy as np
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from torch.onnx import export
import onnx

# --- 1. Загрузка модели ---
model_path = "patchcore_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(model_path, map_location=device)

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
input_size = (224, 224)
batch_size = 1
dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)

# --- 3. Определяем выходы ---
print("🔍 Проверяем, что возвращает модель...")
with torch.no_grad():
    test_output = model(dummy_input)
    if isinstance(test_output, tuple):
        num_outputs = len(test_output)
        output_names = [f"output_{i}" for i in range(num_outputs)]
        dynamic_axes = {'input': {0: 'batch_size'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}
    else:
        output_names = ["output"]
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    print(f"   Имена выходов: {output_names}")

# --- 4. Экспорт в FP32 ONNX ---
onnx_fp32_path = "patchcore_model_fp32.onnx"

print("⏳ Экспорт в FP32 ONNX...")
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
        external_data=False
    )
    size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"✅ FP32 модель сохранена: {onnx_fp32_path} (размер: {size_mb:.2f} МБ)")
except Exception as e:
    print(f"❌ Ошибка при экспорте FP32: {e}")
    exit(1)

# --- 5. Конвертация FP32 → FP16 ---
onnx_fp16_path = "patchcore_model_fp16.onnx"

print("⏳ Конвертация в FP16...")

# Проверяем, существует ли FP32 модель
if not os.path.exists(onnx_fp32_path):
    print(f"❌ FP32 модель не найдена: {onnx_fp32_path}")
    exit(1)

# Пробуем разные методы конвертации
conversion_success = False

# Метод 1: через onnxruntime-tools
try:
    print("   Попытка 1: onnxruntime.tools.float16_conversion...")
    from onnxruntime.tools import float16_conversion
    
    model_fp32 = onnx.load(onnx_fp32_path)
    model_fp16 = float16_conversion.convert_float_to_float16(
        model_fp32,
        keep_io_types=True
    )
    onnx.save(model_fp16, onnx_fp16_path)
    
    # Проверяем, что файл создан
    if os.path.exists(onnx_fp16_path):
        onnx.checker.check_model(model_fp16)
        print(f"   ✅ FP16 модель сохранена: {onnx_fp16_path}")
        conversion_success = True
    else:
        print("   ❌ Файл не создан")
        
except ImportError as e:
    print(f"   ⚠️ onnxruntime-tools не установлен: {e}")
    print("   Установите: pip install onnxruntime-tools")
    
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Метод 2: через onnxconverter-common (если первый не сработал)
if not conversion_success:
    try:
        print("   Попытка 2: onnxconverter_common.float16...")
        from onnxconverter_common import float16
        
        model_fp32 = onnx.load(onnx_fp32_path)
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
        onnx.save(model_fp16, onnx_fp16_path)
        
        if os.path.exists(onnx_fp16_path):
            onnx.checker.check_model(model_fp16)
            print(f"   ✅ FP16 модель сохранена: {onnx_fp16_path}")
            conversion_success = True
        else:
            print("   ❌ Файл не создан")
            
    except ImportError as e:
        print(f"   ⚠️ onnxconverter-common не установлен: {e}")
        print("   Установите: pip install onnxconverter-common")
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

# Метод 3: ручная конвертация (если предыдущие не сработали)
if not conversion_success:
    try:
        print("   Попытка 3: ручная конвертация...")
        model_fp32 = onnx.load(onnx_fp32_path)
        
        # Ручная конвертация всех тензоров в FP16
        from onnx import helper, numpy_helper, TensorProto
        
        # Создаём новый граф с FP16
        new_nodes = []
        for node in model_fp32.graph.node:
            new_nodes.append(node)
        
        # Конвертируем initializers
        new_initializers = []
        for init in model_fp32.graph.initializer:
            if init.data_type == TensorProto.FLOAT:
                # Конвертируем в FP16
                float_data = numpy_helper.to_array(init)
                float16_data = float_data.astype(np.float16)
                new_init = numpy_helper.from_array(float16_data, init.name)
                new_initializers.append(new_init)
            else:
                new_initializers.append(init)
        
        # Создаём новую модель
        new_graph = helper.make_graph(
            new_nodes,
            model_fp32.graph.name,
            model_fp32.graph.input,
            model_fp32.graph.output,
            new_initializers
        )
        
        new_model = helper.make_model(new_graph)
        new_model.opset_import.extend(model_fp32.opset_import)
        
        onnx.save(new_model, onnx_fp16_path)
        
        if os.path.exists(onnx_fp16_path):
            print(f"   ✅ FP16 модель сохранена (ручная конвертация): {onnx_fp16_path}")
            conversion_success = True
        else:
            print("   ❌ Файл не создан")
            
    except Exception as e:
        print(f"   ❌ Ошибка ручной конвертации: {e}")

# --- 6. Проверка результатов ---
if conversion_success and os.path.exists(onnx_fp16_path):
    size_fp32 = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    size_fp16 = os.path.getsize(onnx_fp16_path) / (1024 * 1024)
    
    print(f"\n📊 Сравнение размеров:")
    print(f"   FP32: {size_fp32:.2f} МБ")
    print(f"   FP16: {size_fp16:.2f} МБ")
    print(f"   Сжатие: {(1 - size_fp16 / size_fp32) * 100:.1f}%")
    
    # --- 7. Проверка FP16 модели ---
    print("\n🔍 Проверка FP16 модели...")
    try:
        import onnxruntime as ort
        
        ort_session = ort.InferenceSession(onnx_fp16_path, providers=['CPUExecutionProvider'])
        
        input_name = ort_session.get_inputs()[0].name
        test_input = np.random.randn(batch_size, 3, input_size[0], input_size[1]).astype(np.float32)
        
        ort_outputs = ort_session.run(None, {input_name: test_input})
        
        print(f"✅ Тест FP16 модели успешен")
        print(f"  Количество выходов: {len(ort_outputs)}")
        for i, out in enumerate(ort_outputs):
            print(f"  Выход {i}: форма {out.shape}, тип {out.dtype}")
        
    except Exception as e:
        print(f"⚠️ Ошибка при проверке FP16 модели: {e}")
else:
    print("\n❌ Не удалось создать FP16 модель")
    print("📦 Установите необходимые библиотеки:")
    print("   pip install onnxruntime-tools")
    print("   или")
    print("   pip install onnxconverter-common")

print("\n📁 Доступные файлы:")
for file in os.listdir("."):
    if file.endswith(".onnx") or file.endswith(".data"):
        size = os.path.getsize(file) / (1024 * 1024)
        print(f"   {file}: {size:.2f} МБ")

print("\n🎉 Скрипт завершен!")
