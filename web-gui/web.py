import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

import av
import cv2
import torch
from einops import rearrange
import numpy as np

import time



classes = {
  0: "background",
  1: "bottled soda",
  2: "cheese",
  3: "chocolate bar",
  4: "cofee",
  5: "condensed milk",
  6: "cooking oil",
  7: "corned beef",
  8: "garlic",
  9: "instant noodles",
  10: "ketchup",
  11: "lemon",
  12: "all purpose cream",
  13: "mayonnaise",
  14: "peanut butter",
  15: "pasta",
  16: "pineapple juice",
  17: "crackers",
  18: "sardines",
  19: "shampoo",
  20: "soap",
  21: "soy sauce",
  22: "toothpaste",
  23: "canned tuna",
  24: "ethyl alcohol"    
}

st.title("Grocery Items Detection and Segmentation", "center")

with st.container():
    def video_frame_callback(frame):
        start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        img_tensor = torch.from_numpy(img).float()
        img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")

        # inference
        preds = model.predict(img_tensor)
        for result in preds:
            #print(result.names[result.boxes.cls])
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
            if task == "segmentation":
                for mask in result.masks.xy:
                    # create polygon from a series of mask points
                    points = np.int32([mask])
                    # print(points)
                    cv2.fillPoly(img, points, (0, 255, 0))


        # draw bounding boxes on the image
        # how to display fps 
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
            _model = st.selectbox("Choose Model", ["yolo11n-seg"])

    models = {
        "detection": {
            "yolo11m-finetuned": "/mnt/c/Users/Acer/Graduate/Courses/AI 222/grocery-cv/web-gui/grocery-detect-yolo11m-best.pt",
            "yolo11n-finetuned": "/mnt/c/Users/Acer/Graduate/Courses/AI 222/grocery-cv/web-gui/grocery-detect-yolo11n-best.pt",
            "yolo11m": "/mnt/c/Users/Acer/Graduate/Courses/AI 222/grocery-cv/web-gui/yolo11m.pt"
        },
        "segmentation": {
            "yolo11n-seg": "/mnt/c/Users/Acer/Graduate/Courses/AI 222/grocery-cv/web-gui/yolo11n-seg.pt",
        }
    }

# with col2:
    model = YOLO(models[task][_model])
    webrtc_streamer(key="sample",video_frame_callback=video_frame_callback)