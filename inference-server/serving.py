# create a flask server that accepts video frames and returns the processed frames
from flask import Flask, request, Response
import cv2

import numpy as np
import torch
from einops import rearrange

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 16 MB limit

from ultralytics import YOLO

_detect_model_path = '/data/students/gerry/repos/grocery-cv/aiml/outputs/2024-11-15/08-39-03/grocery-cv-detect-yolon-complete-data/train/weights/best.pt'
_segment_model_path = '/data/students/gerry/repos/grocery-cv/aiml/outputs/2024-11-14/00-45-20/grocery-cv-seg-yolon-complete-data/train/weights/best.pt'

detect_model = YOLO(_detect_model_path)  # Replace with your model path if different
segment_model = YOLO(_segment_model_path)  # Replace with your model path if different


from custom_classes import classes

def process_frame(task, frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_tensor = torch.from_numpy(frame).float()
    img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")
    model = detect_model if task == "detect" else segment_model
    preds = model.predict(img_tensor, device='1')

    for result in preds:
        #print(result.names[result.boxes.cls])
        for box in result.boxes:
            if _detect_model_path in ["yolo11m", "yolo11n-seg"]:
                cls = result.names[box.cls.item()]
            else:
                cls = classes[box.cls.item()]
            conf = box.conf.item()
            # print(cls, conf, model.device, img_tensor.device)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)    
            # add mask xy to image

            if task == "segmentation" and result.masks is not None:
                for mask in result.masks.xy:
                    # create polygon from a series of mask points
                    points = np.int32([mask])
                    # print(points)
                    print(points.shape)
                    cv2.fillPoly(img, points, (0, 255, 0))


    frame = cv2.putText(frame, "Processed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

@app.route('/health', methods=['GET'])
def health():
    return "Healthy", 200

@app.route('/infer/<task>', methods=['POST'])
def process_video_frame():
    if 'frame' not in request.files:
        return "No frame provided", 400

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    match task:
        case "detect":
            processed_frame = process_frame(detect_model, frame)
        case "segment":
            processed_frame = process_frame(segment_model, frame)
        case _:
            processed_frame = process_frame(detect_model, frame)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    response = Response(buffer.tobytes(), mimetype='image/jpeg')

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)