from PIL import Image
from ultralytics import YOLO

# 加载一个在COCO预训练的YOLOv8n模型
model = YOLO('yolov8n.pt')

# 显示模型信息（可选）
model.info()

# 验证模型
metrics = model.val()  # 无需参数，数据集和设置记忆
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # 包含每个类别的map50-95列表

# 使用YOLOv8n模型在'bus.jpg'图片上运行推理
