from PIL import Image
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.info()

results = model.train(data='VisDrone.yaml', epochs=20, device=[0,1,2,3])
