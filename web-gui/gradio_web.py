import gradio as gr
import numpy as np
import cv2


from ultralytics import YOLO

import av
import cv2
import torch
from einops import rearrange
import numpy as np

import time

import json
file_path = 'model_paths.json'
with open(file_path, 'r') as file:
    paths = json.load(file)
with open('classes.json', 'r') as file:
    classes = json.load(file)


model_detect = YOLO(paths['detect'], task='detect')  # Replace with your model path if different
model_segment = YOLO(paths['segment'], task='segment')  # Replace with your model path if different

def video_frame_callback(img, task):
    start_time = time.time()
    print(img.shape)
    img_size = (768,1280)
    img = cv2.resize(img, img_size)
    img_tensor = torch.from_numpy(img).float()
    img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")

    model = model_detect if task == "detection" else model_segment
    _model = paths[task]
    # inference
    preds = model.predict(img_tensor)
    for result in preds:
        #print(result.names[result.boxes.cls])
        if result.boxes is not None:
            for box in result.boxes:
                if _model in ["yolo11m", "yolo11n-seg"]:
                    cls = result.names[str(int(box.cls.item()))]
                else:
                    cls = classes[str(int(box.cls.item()))]
                conf = box.conf.item()
                # print(cls, conf, model.device, img_tensor.device)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{str(int(box.cls.item()))} {cls} {conf:.2f}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # add mask xy to image
        if task == "segment" and result.masks is not None:
            for mask in result.masks.xy:
                # create polygon from a series of mask points
                points = np.int32([mask])
                # print(points)
                print(points.shape)
                cv2.fillPoly(img, points, (0, 255, 0))

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    img = cv2.resize(img, (1280, 720))
    return img


# css=""".my-group {max-width: 800px !important; max-height: 600px !important;}
#             .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Group():
            input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)
            task = gr.Dropdown(choices=["detect", "segment"],
                                    value="detect", label="Task")
    input_img.stream(video_frame_callback, [input_img, task], [input_img], time_limit=30, stream_every=0.1)


demo.launch(server_name="0.0.0.0", server_port=3333)