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
from custom_classes import classes
from model_maps import models

paths = {
    'detect': '/data/students/gerry/repos/grocery-cv/aiml/outputs/2024-11-15/08-39-03/grocery-cv-detect-yolon-complete-data/train/weights/best.pt',
    'segment': '/data/students/gerry/repos/grocery-cv/aiml/outputs/2024-11-14/00-45-20/grocery-cv-seg-yolon-complete-data/train/weights/best.pt'
}


model_detect = YOLO(paths['detect'])  # Replace with your model path if different
model_segment = YOLO(paths['segment'])  # Replace with your model path if different

def video_frame_callback(img, task):
    start_time = time.time()
    #img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (768,1280))
    img_tensor = torch.from_numpy(img).float()
    img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")

    model = model_detect if task == "detection" else model_segment
    _model = paths[task]
    # inference
    preds = model.predict(img_tensor, device='0')
    for result in preds:
        #print(result.names[result.boxes.cls])
        if result.boxes is not None:
            for box in result.boxes:
                if _model in ["yolo11m", "yolo11n-seg"]:
                    cls = result.names[box.cls.item()]
                else:
                    cls = classes[box.cls.item()]
                conf = box.conf.item()
                # print(cls, conf, model.device, img_tensor.device)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{cls} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # img = cv2.resize(img, (720,1280))
    return img
    #return av.VideoFrame.from_ndarray(img, format="bgr24")


css=""".my-group {max-width: 600px !important; max-height: 600px !important;}
            .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            task = gr.Dropdown(choices=["detect", "segment"],
                                    value="detect", label="Task")
            input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)
    input_img.stream(video_frame_callback, [input_img, task], [input_img], time_limit=30, stream_every=0.1)


demo.launch(server_name="0.0.0.0", server_port=5001)