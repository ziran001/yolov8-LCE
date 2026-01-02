from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型（可选n/s/m/l/x）
    model = YOLO("yolov8n.pt")
    # 开始训练
    model.train(
        data="datasets\data.yaml",  # 数据集配置文件
        epochs=3,
        imgsz=640,
        batch=4,
        workers=0,
    )
