import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
from twilio.rest import Client
from aiortc.mediastreams import MediaStreamTrack

class CustomMediaStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()

    async def recv(self, frame):
        print(frame)
        return frame

import av
import cv2
import torch
from einops import rearrange
import numpy as np
import os

import time

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

st.title("Grocery Items Detection and Segmentation", "center")

from custom_classes import classes
from model_maps import models

with st.container():
    def video_frame_callback(frame):

        return frame
        start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        img_tensor = torch.from_numpy(img).float()
        img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")

        # inference
        preds = model.predict(img_tensor)
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
            if task == "segmentation" and result.masks is not None:
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

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    task = st.selectbox("Choose Task", ["detection", "segmentation"])

    # add streamlit selector for detection model
    match task:
        case "detection":
            _model = st.selectbox("Choose Model", ["yolo11m-finetuned", "yolo11n-finetuned", "yolo11m"])
        case "segmentation":
            _model = st.selectbox("Choose Model", ["yolo11n-seg", "yolo11m-seg-finetuned"])

# with col2:
    # model = YOLO(models[task][_model])
    # st.write(_model)
    account_sid = 'AC43112e7058999c50a8fc861e9ee02873' #os.environ['AC43112e7058999c50a8fc861e9ee02873']
    auth_token = '82afe3b6f1bebc442f4aacce6357861b' #os.environ['82afe3b6f1bebc442f4aacce6357861b']
    client = Client(account_sid, auth_token)
    token = client.tokens.create()


    webrtc_streamer(
        key="sample",
        # mode=WebRtcMode.RECVONLY,
        video_frame_callback=video_frame_callback,
        # rtc_configuration={  # Add this config
        #     "iceServers": token.ice_servers
        # },
        # source_video_track=CustomMediaStreamTrack(),
    )