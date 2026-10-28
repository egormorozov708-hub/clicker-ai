"""Автоматическая разметка новых скриншотов обученной моделью.

Запуск:
    python auto_label.py --model best.pt --source screenshots --out auto_labels

Для каждой картинки без .json создаётся:
  * auto_labels/<имя>.json  — формат X-AnyLabeling (можно открыть и проверить);
  * auto_labels/<имя>.txt   — формат YOLO (class xc yc w h).

По умолчанию сохраняются все классы модели. Ограничить набор можно так:
    python auto_label.py --model best.pt --classes game,upgrade

Дальше разметку можно открыть в X-AnyLabeling, поправить и экспортировать.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def shape_from_box(box: list[float], label: str, conf: float) -> dict:
    x1, y1, x2, y2 = box
    return {
        "label": label,
        "score": round(float(conf), 4),
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def to_anylabeling(image_path: Path, shapes: list[dict], w: int, h: int) -> dict:
    return {
        "version": "4.1.1",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "description": "auto-labeled",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--source", default="screenshots")
    parser.add_argument("--out", default="auto_labels")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--classes",
        default="",
        help="через запятую; пусто = все классы модели",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    allowed = {c.strip() for c in args.classes.split(",") if c.strip()}
    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    images = sorted(
        p for p in source.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    done = 0
    for img_path in images:
        json_path = img_path.with_suffix(".json")
        if json_path.exists() and not args.overwrite:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        result = model(str(img_path), imgsz=640, conf=args.conf, verbose=False)[0]
        shapes: list[dict] = []
        lines: list[str] = []
        for box in sorted(result.boxes, key=lambda b: float(b.conf), reverse=True):
            label = model.names[int(box.cls)]
            if allowed and label not in allowed:
                continue
            xyxy = [float(v) for v in box.xyxy[0]]
            shapes.append(shape_from_box(xyxy, label, float(box.conf)))
            x1, y1, x2, y2 = xyxy
            lines.append(
                f"{int(box.cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        if not shapes:
            print(f"[skip] {img_path.name}: объекты не найдены")
            continue
        (out / f"{img_path.stem}.json").write_text(
            json.dumps(to_anylabeling(img_path, shapes, w, h), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        done += 1
        labels = ", ".join(
            f"{s['label']} {s['score']:.2f}" for s in shapes
        )
        print(f"[ok] {img_path.name}: {labels}")

    print(f"Готово: размечено {done} изображений -> {out}")


if __name__ == "__main__":
    main()
