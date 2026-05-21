import cv2
import sys
import json
import numpy as np
import time
import openvino as ov
import openvino.properties.hint as hints
from collections import deque

# ------------------- Загрузка меток классов из JSON -------------------
def load_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get('id2label', {})

# ------------------- Класс для асинхронного инференса с колбэками -------------------
class AsyncOpenVinoYolov8:
    __slots__ = ('confidence_threshold', 'infer_queue', 'input_layer', 'output_layer',
                 'input_height', 'input_width', 'num_requests', 'results_queue')
    def __init__(self, model_path, conf_thresh, device='CPU', num_requests=2):
        self.confidence_threshold = conf_thresh
        core = ov.Core()
        model = core.read_model(model_path)
        compiled = core.compile_model(model, device, config={hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
        self.infer_queue = ov.AsyncInferQueue(compiled, jobs=num_requests)
        self.input_layer = compiled.input(0)
        self.output_layer = compiled.output(0)
        shape = self.input_layer.shape
        self.input_height, self.input_width = shape[2], shape[3]
        self.num_requests = num_requests
        # Очередь для результатов (будет заполняться из колбэков)
        self.results_queue = deque()

    def preprocess(self, image):
        resized = cv2.resize(image, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
        return blob

    def postprocess(self, raw_output, original_shape):
        pred = np.squeeze(raw_output, axis=0).transpose(1, 0)   # (8400,16)
        boxes = pred[:, :4]
        scores = np.max(pred[:, 4:], axis=1)
        class_ids = np.argmax(pred[:, 4:], axis=1)
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return np.empty((0, 4)), np.empty(0), np.empty(0)
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        h_img, w_img = original_shape
        sx = w_img / self.input_width
        sy = h_img / self.input_height
        xyxy = []
        for cx, cy, w, h in boxes:
            x1 = (cx - w/2) * sx
            y1 = (cy - h/2) * sy
            x2 = (cx + w/2) * sx
            y2 = (cy + h/2) * sy
            xyxy.append([max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)])
        return np.array(xyxy), scores, class_ids

    # Колбэк, который будет вызываться по завершении каждого запроса
    def _callback(self, request, user_data):
        # user_data = (frame_index, original_frame)
        idx, orig_frame = user_data
        raw_out = request.get_output_tensor(self.output_layer).data
        boxes, scores, class_ids = self.postprocess(raw_out, orig_frame.shape[:2])
        # Сохраняем результат в общую очередь вместе с индексом для сохранения порядка
        self.results_queue.append((idx, orig_frame, boxes, scores, class_ids))

    def predict_async(self, frame, frame_idx):
        """Запускает асинхронный инференс, передавая колбэк."""
        blob = self.preprocess(frame)
        request_id = self.infer_queue.get_idle_request_id()
        request = self.infer_queue[request_id]
        # Устанавливаем колбэк, передаём ID кадра и само изображение
        request.set_callback(self._callback, (frame_idx, frame))
        request.start_async({self.input_layer: blob})
        return request_id

    def wait_all(self):
        self.infer_queue.wait_all()

# ------------------- Генерация цветов -------------------
def get_color(class_id):
    np.random.seed(class_id)
    return tuple(map(int, np.random.randint(0, 255, 3)))

# ------------------- MAIN -------------------
if __name__ == '__main__':
    if len(sys.argv) < 9:
        print("Usage: python script.py <model.onnx> <input.mp4> <output.mp4> "
              "<slice_h> <slice_w> <overlap_h> <overlap_w> <conf_thresh> [labels.json] [num_requests]")
        print("Example: python script.py model.onnx in.mp4 out.mp4 640 640 0.2 0.2 0.25 labels.json 4")
        sys.exit(1)

    model_path = sys.argv[1]
    input_video = sys.argv[2]
    output_video = sys.argv[3]
    conf_thresh = float(sys.argv[8])
    labels_path = sys.argv[9] if len(sys.argv) > 9 else None
    num_requests = int(sys.argv[10]) if len(sys.argv) > 10 else 2

    if labels_path and labels_path.endswith('.json'):
        id2label = load_labels(labels_path)
    else:
        id2label = {
            "0": "LightVehicle", "1": "Person", "2": "Building", "3": "UPole",
            "4": "Boat", "5": "Bike", "6": "Container", "7": "Truck/Bus",
            "8": "Gastank", "9": "Digger", "10": "SolarPanels", "11": "Truck/Bus"
        }
    print(f"Loaded {len(id2label)} classes. Confidence threshold = {conf_thresh}")
    print(f"Using AsyncInferQueue with {num_requests} parallel requests (THROUGHPUT mode, callbacks)")

    model = AsyncOpenVinoYolov8(model_path, conf_thresh=conf_thresh, device='CPU', num_requests=num_requests)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: cannot open video")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    frame_idx = 0
    sent_count = 0
    next_expected_idx = 0
    inference_times = []
    prev_time = time.time()
    fps_display = 0
    total_objects = 0

    print("Processing video asynchronously with callbacks...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Отправляем кадр на инференс (не ждём результата)
        model.predict_async(frame, frame_idx)
        sent_count += 1
        frame_idx += 1

        # Проверяем, не пришли ли уже результаты из колбэков
        while model.results_queue and model.results_queue[0][0] == next_expected_idx:
            idx, orig_frame, boxes, scores, class_ids = model.results_queue.popleft()
            # Замеряем только время постобработки (для статистики)
            start = time.time()
            # постобработка уже выполнена в колбэке, просто рисуем
            infer_time = time.time() - start
            inference_times.append(infer_time)

            num_obj = len(boxes)
            total_objects += num_obj

            # Отрисовка
            annotated = orig_frame.copy()
            for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                color = get_color(cid)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{id2label.get(str(cid), str(cid))}: {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(5, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            now = time.time()
            delta = now - prev_time
            if delta > 0:
                fps_display = 1.0 / delta
            prev_time = now

            cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Objects: {num_obj}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(annotated)
            next_expected_idx += 1

            if next_expected_idx % 30 == 0 and inference_times:
                avg_time = np.mean(inference_times[-30:]) * 1000
                print(f"Frame {next_expected_idx}/{total_frames} | FPS: {fps_display:.1f} | "
                      f"Avg postproc: {avg_time:.1f} ms | Conf thresh: {conf_thresh}", end='\r')

        # Не даём очереди запросов переполниться (это ограничение можно подстроить)
        if sent_count - next_expected_idx > num_requests * 2:
            # Если слишком много необработанных запросов, ждём немного
            time.sleep(0.001)

    # Дожидаемся всех оставшихся запросов
    model.wait_all()
    # Обрабатываем оставшиеся результаты (должны быть все)
    while model.results_queue:
        idx, orig_frame, boxes, scores, class_ids = model.results_queue.popleft()
        num_obj = len(boxes)
        total_objects += num_obj
        annotated = orig_frame.copy()
        for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            color = get_color(cid)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{id2label.get(str(cid), str(cid))}: {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(5, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out.write(annotated)
        next_expected_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n\n" + "="*60)
    print(f"Frames processed: {next_expected_idx}")
    print(f"Total objects: {total_objects}")
    print(f"Avg objects/frame: {total_objects/next_expected_idx:.2f}" if next_expected_idx else "N/A")
    print(f"Avg postprocessing time: {np.mean(inference_times)*1000:.1f} ms" if inference_times else "N/A")
    print(f"Overall FPS: {next_expected_idx / (time.time() - prev_time + 1e-6):.1f}")
    print("="*60)
    print(f"Output saved to: {output_video}")
