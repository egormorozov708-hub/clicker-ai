"""Обучение итоговой модели: game + upgrade + advertising + bonus.

Запуск:
    python train_final.py

Результат: runs/detect/clicker_final/weights/best.pt
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data_final.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        workers=0,
        device="cpu",
        patience=20,
        name="clicker_final",
        exist_ok=True,
        plots=True,
        seed=0,
        cache="ram",
        amp=False,
    )
