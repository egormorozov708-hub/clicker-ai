"""Экспорт итоговой модели в ONNX для бота и авторазметки в X-AnyLabeling.

Запуск:
    python ptToOnnx.py
    python ptToOnnx.py --weights runs/detect/clicker_levels/weights/best.pt --out best.onnx

Результат: best.onnx в корне проекта. Именно этот файл использует app.py
и указывается в my_cookie_model.yaml.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт весов Clicker AI в ONNX")
    parser.add_argument(
        "--weights",
        default="runs/detect/clicker_levels/weights/best.pt",
        help="путь к итоговым весам .pt",
    )
    parser.add_argument("--out", default="best.onnx", help="куда скопировать ONNX")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Файл модели не найден: {weights}")

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=False,
            half=False,
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out.resolve():
        shutil.copy2(exported, out)
        print(f"ONNX скопирован: {exported} -> {out}")
    else:
        print(f"ONNX сохранён: {out}")


if __name__ == "__main__":
    main()
