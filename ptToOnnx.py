"""Экспорт обученной модели в ONNX для авторазметки в X-AnyLabeling.

Запуск:
    python ptToOnnx.py
    python ptToOnnx.py --weights best.pt --imgsz 640

Результат: best.onnx рядом с весами. Этот файл указывается в my_cookie_model.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Файл модели не найден: {weights}")

    model = YOLO(str(weights))
    output = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f"ONNX сохранён: {output}")


if __name__ == "__main__":
    main()
