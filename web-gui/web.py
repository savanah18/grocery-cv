from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

import av
import cv2


import torch
from einops import rearrange
import time

model = YOLO("yolo11m.pt")
# model.eval()


def video_frame_callback(frame):
    start_time = time.time()

    img = frame.to_ndarray(format="bgr24")
    img_tensor = torch.from_numpy(img).float()
    img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")

    # inference
    preds = model(img_tensor)
    for result in preds:
        #print(result.names[result.boxes.cls])
        for box in result.boxes:
            cls = result.names[box.cls.item()]
            conf = box.conf.item()
            # print(cls, conf, model.device, img_tensor.device)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # draw bounding boxes on the image
    # how to display fps 
    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="sample",video_frame_callback=video_frame_callback)