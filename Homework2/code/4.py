from PIL import Image
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.info()

# results = model.train(data='VisDrone.yaml', epochs=20, device=[0, 1, 2, 3])

# results = model.train(data='VisDrone.yaml', epochs=20, cls=0.7, dfl=1.7, box=8.0, device=0)
# results = model.train(data='VisDrone.yaml', epochs=50, cls=0.7, dfl=1.7, box=8.0, device=1)

results = model.train(data='VisDrone.yaml', epochs=20, lr0=8e-3, lrf=5e-3, imgsz=640, device=2)
# results = model.train(data='VisDrone.yaml', epochs=50, lr0=8e-3, lrf=5e-3, imgsz=640, device=3)

# results = model.train(data='VisDrone.yaml', epochs=20, cls=0.7, dfl=1.7, box=8.0, lr0=8e-3, lrf=5e-3, imgsz=640, device=1)
# results = model.train(data='VisDrone.yaml', epochs=50, cls=0.7, dfl=1.7, box=8.0, lr0=8e-3, lrf=5e-3, imgsz=640, device=1)
# results = model.train(data='VisDrone.yaml', epochs=50, cls=0.4, dfl=1.3, box=7.0, lr0=8e-3, lrf=5e-3, imgsz=640, device=1)
