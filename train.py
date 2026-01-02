from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型（可选n/s/m/l/x）
    model = YOLO("ultralytics/cfg/models/v8/yolov8-lce.yaml")
    # 开始训练
    model.train(
        data="data/pest24.yaml",  # 数据集配置文件
        epochs=100,
        imgsz=640,
        batch=32,
        workers=8,
        wiouv3=True,
        wiou_alpha=1.7,
        wiou_delta=2.7,
    )
