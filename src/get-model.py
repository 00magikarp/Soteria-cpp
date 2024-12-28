from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("../include/model/yolo11n.pt")
    model.export(format="onnx")
