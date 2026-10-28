"""Дообучение итоговой модели на двух уровнях и рекламе.

Старт — текущая модель best.pt (4 класса), затем fine-tune на объединённом
датасете: старые кадры + level 2 + advertising.

Запуск:
    python train_final.py

Результат: runs/detect/clicker_levels/weights/best.pt
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("best.pt")
    model.train(
        data="data_final.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        workers=0,
        device="cpu",
        patience=12,
        name="clicker_levels",
        exist_ok=True,
        plots=True,
        seed=0,
        cache="ram",
        amp=False,
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
    )
