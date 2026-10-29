"""Дообучение итоговой модели на двух уровнях и рекламе.

Скрипт выполняет fine-tune стартовых весов на объединённом датасете
(старые кадры + level 2 + advertising). Датасет и промежуточные веса в
репозиторий не входят: подготовьте их локально командами из README.

Запуск:
    python train_final.py --weights best.pt
    python train_final.py --weights bestV1.pt --epochs 40

Результат: runs/detect/clicker_levels/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune итоговой модели Clicker AI")
    parser.add_argument(
        "--weights",
        default="best.pt",
        help="стартовые веса (промежуточная модель, локальный файл)",
    )
    parser.add_argument("--data", default="data_final.yaml", help="конфиг датасета")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 0,1 ...")
    parser.add_argument("--name", default="clicker_levels")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Стартовые веса не найдены: {weights}. "
            "Укажите --weights путь/к/предыдущей/best.pt."
        )

    model = YOLO(str(weights))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=args.device,
        patience=12,
        name=args.name,
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


if __name__ == "__main__":
    main()
