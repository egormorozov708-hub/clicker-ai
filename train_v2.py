"""Обучение детектора игрового персонажа (предварительная модель).

Запуск:
    python train_v2.py

Результат: runs/detect/game_v1/weights/best.pt
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data_v2.yaml",
        epochs=60,
        imgsz=640,
        batch=8,
        workers=0,
        device="cpu",
        patience=20,
        name="game_v1",
        exist_ok=True,
        plots=True,
        seed=0,
    )
