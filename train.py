"""Базовый пример обучения детектора (упрощённый вариант из задания).

Запуск:
    python train.py

Для итоговой модели используется train_final.py (60 эпох, 4 класса).
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        workers=0,
        device="cpu",
        patience=15,
        name="clicker_base",
        exist_ok=True,
        seed=0,
    )
