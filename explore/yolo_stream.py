# import cv2

# # Open a connection to the camera
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     # show the frame
#     cv2.imshow('frame', frame)

# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11m.pt", device='gpu')
results = model(source=0, show=True) 