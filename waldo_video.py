import cv2
import sys
import json
import numpy as np
import time
import openvino as ov

def load_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get('id2label', {})

class SmoothDetections:
    __slots__ = ('alpha', 'iou_threshold', 'max_missing', 'tracks')
    def __init__(self, alpha=0.3, iou_threshold=0.33, max_missing=5):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks = []

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections, scores, class_ids):
        if len(detections) == 0:
            for t in self.tracks:
                t[3] += 1
            self.tracks = [t for t in self.tracks if t[3] < self.max_missing]
            return np.empty((0, 4)), np.empty(0), np.empty(0)

        new_tracks = []
        used_det = set()

        for i, (det_box, det_score, det_cls) in enumerate(zip(detections, scores, class_ids)):
            best_iou = 0.0
            best_idx = -1
            for j, (trk_box, trk_cls, trk_score, miss) in enumerate(self.tracks):
                iou_val = self.iou(det_box, trk_box)
                if iou_val > best_iou and iou_val > self.iou_threshold:
                    best_iou = iou_val
                    best_idx = j
            if best_idx >= 0:
                old_box = self.tracks[best_idx][0]
                new_box = self.alpha * det_box + (1 - self.alpha) * old_box
                self.tracks[best_idx][0] = new_box
                self.tracks[best_idx][1] = det_cls
                self.tracks[best_idx][2] = det_score
                self.tracks[best_idx][3] = 0
                new_tracks.append(self.tracks[best_idx])
                used_det.add(i)
            else:
                new_tracks.append([det_box.copy(), det_cls, det_score, 0])

        for j, track in enumerate(self.tracks):
            if j not in used_det:
                new_tracks.append([track[0], track[1], track[2], track[3] + 1])

        self.tracks = [t for t in new_tracks if t[3] < self.max_missing]

        active_boxes, active_scores, active_classes = [], [], []
        for track in self.tracks:
            if track[3] == 0:
                active_boxes.append(track[0])
                active_scores.append(track[2])
                active_classes.append(track[1])
        return (np.array(active_boxes) if active_boxes else np.empty((0, 4)),
                np.array(active_scores) if active_scores else np.empty(0),
                np.array(active_classes) if active_classes else np.empty(0))

DEFAULT_CLASS_THRESHOLDS = {
    "0": 0.2,      # LightVehicle
    "1": 0.5,      # Person
    "2": 0.4,      # Building
    "3": 0.25,     # UPole
    "4": 0.35,     # Boat
    "5": 0.6,      # Bike
    "6": 0.4,      # Container
    "7": 0.3,     # Truck
    "8": 0.45,     # Gastank
    "9": 0.4,      # Digger
    "10": 0.3,     # SolarPanels
    "11": 0.45,     # Bus
}
class OpenVinoYolov8:
    __slots__ = ('confidence_threshold', 'model', 'input_layer', 'output_layer',
                 'input_height', 'input_width')
    def __init__(self, model_path, conf_thresh, device='CPU'):
        self.confidence_threshold = conf_thresh
        core = ov.Core()
        model = core.read_model(model_path)
        self.model = core.compile_model(model, device)
        self.input_layer = self.model.input(0)
        self.output_layer = self.model.output(0)
        shape = self.input_layer.shape
        self.input_height, self.input_width = shape[2], shape[3]

    def predict(self, image):
        resized = cv2.resize(image, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
        outputs = self.model.infer_new_request({self.input_layer: blob})
        raw = outputs[self.output_layer]
        pred = np.squeeze(raw, axis=0).transpose(1, 0)
        boxes = pred[:, :4]
        scores = np.max(pred[:, 4:], axis=1)
        class_ids = np.argmax(pred[:, 4:], axis=1)
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return np.empty((0, 4)), np.empty(0), np.empty(0)
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        h_img, w_img = image.shape[:2]
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


# ------------------- Генерация цветов -------------------
def get_color(class_id):
    np.random.seed(class_id)
    return tuple(map(int, np.random.randint(0, 255, 3)))


# ------------------- MAIN -------------------
if __name__ == '__main__':
    if len(sys.argv) < 9:
        print("Usage: python script.py <model.onnx> <input.mp4> <output.mp4> "
              "<slice_h> <slice_w> <overlap_h> <overlap_w> <conf_thresh> [labels.json] [alpha] [iou_thresh]")
        print("Example: python script.py model.onnx in.mp4 out.mp4 640 640 0.2 0.2 0.25 labels.json 0.3 0.4")
        sys.exit(1)

    model_path = sys.argv[1]
    input_video = sys.argv[2]
    output_video = sys.argv[3]
    conf_thresh = float(sys.argv[8])       
    labels_path = sys.argv[9] if len(sys.argv) > 9 else None
    alpha = float(sys.argv[10]) if len(sys.argv) > 10 else 0.3
    iou_thresh = float(sys.argv[11]) if len(sys.argv) > 11 else 0.4

    # Загрузка меток
    if labels_path and labels_path.endswith('.json'):
        id2label = load_labels(labels_path)
    else:
        id2label = {
            "0": "LightVehicle", "1": "Person", "2": "Building", "3": "UPole",
            "4": "Boat", "5": "Bike", "6": "Container", "7": "Truck/Bus",
            "8": "Gastank", "9": "Digger", "10": "SolarPanels", "11": "Truck/Bus"
        }
    print(f"Loaded {len(id2label)} classes. Confidence threshold = {conf_thresh}")

    model = OpenVinoYolov8(model_path, conf_thresh=conf_thresh, device='CPU')

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

    smoother = SmoothDetections(alpha=alpha, iou_threshold=iou_thresh, max_missing=5)

    frame_cnt = 0
    total_objects = 0
    objects_per_frame = []
    inference_times = []
    prev_time = time.time()
    fps_display = 0

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        boxes, scores, class_ids = model.predict(frame)
        boxes, scores, class_ids = smoother.update(boxes, scores, class_ids)
        inference_times.append(time.time() - start)

        num_obj = len(boxes)
        total_objects += num_obj
        objects_per_frame.append(num_obj)

        now = time.time()
        delta = now - prev_time
        if delta > 0:
            fps_display = 1.0 / delta
        prev_time = now

        annotated = frame.copy()
        for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            color = get_color(cid)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{id2label.get(str(cid), str(cid))}: {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(5, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"Objects: {num_obj}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(annotated)
        frame_cnt += 1

        if frame_cnt % 30 == 0:
            avg_time = np.mean(inference_times[-30:]) * 1000
            print(f"Frame {frame_cnt}/{total_frames} | FPS: {fps_display:.1f} | "
                  f"Avg inf: {avg_time:.1f} ms | Conf thresh: {conf_thresh}", end='\r')

    print("\n\n" + "="*60)
    print(f"Frames processed: {frame_cnt}")
    print(f"Total objects: {total_objects}")
    print(f"Avg objects/frame: {total_objects/frame_cnt:.2f}")
    print(f"Max objects: {max(objects_per_frame) if objects_per_frame else 0}")
    print(f"Avg inference time: {np.mean(inference_times)*1000:.1f} ms")
    print(f"Overall FPS: {frame_cnt / (time.time() - prev_time + 1e-6):.1f}")
    print("="*60)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nOutput saved to: {output_video}")
