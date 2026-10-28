"""Аугментация датасета с пересчётом рамок.

Скрипт увеличивает train-часть датасета в формате YOLO: изображения и txt-файлы
с рамками преобразуются согласованно, поэтому координаты остаются корректными.

Запуск:
    python augment.py --source dataset_final --out dataset_aug --copies 3

Результат:
    dataset_aug/images/train, dataset_aug/labels/train  — оригинал + копии
    dataset_aug/images/val,   dataset_aug/labels/val    — без изменений
    data_aug.yaml                                       — конфиг для обучения

Важно: горизонтальное отражение и поворот на 90 градусов не используются.
Интерфейс игры несимметричен (улучшения слева, реклама справа), поэтому такие
преобразования создали бы нереалистичные кадры и противоречивую разметку.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import albumentations as A
import cv2

ROOT = Path(__file__).resolve().parent


def build_transform() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.18, contrast_limit=0.18, p=0.6
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                    A.GaussNoise(),
                ],
                p=0.25,
            ),
            A.ImageCompression(quality_range=(70, 95), p=0.3),
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.02, 0.02),
                rotate=(-2, 2),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.35,
            clip=True,
        ),
    )


def read_labels(path: Path) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not path.exists():
        return boxes, classes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        classes.append(int(float(parts[0])))
        boxes.append([float(v) for v in parts[1:]])
    return boxes, classes


def write_labels(path: Path, boxes: list[list[float]], classes: list[int]) -> None:
    lines = []
    for cls, (xc, yc, bw, bh) in zip(classes, boxes):
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_split(source: Path, out: Path, split: str) -> int:
    n = 0
    for img in sorted((source / "images" / split).glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(img, out / "images" / split / img.name)
        label = source / "labels" / split / f"{img.stem}.txt"
        if label.exists():
            shutil.copy2(label, out / "labels" / split / label.name)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset_final")
    parser.add_argument("--out", default="dataset_aug")
    parser.add_argument("--copies", type=int, default=3, help="копий на изображение")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = ROOT / args.source
    out = ROOT / args.out
    if not (source / "images" / "train").exists():
        raise SystemExit(f"Не найден датасет: {source}")

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    transform = build_transform()
    transform.set_random_seed(args.seed)

    originals = 0
    created = 0
    for img_path in sorted((source / "images" / "train").glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        boxes, classes = read_labels(source / "labels" / "train" / f"{img_path.stem}.txt")

        shutil.copy2(img_path, out / "images" / "train" / img_path.name)
        write_labels(out / "labels" / "train" / f"{img_path.stem}.txt", boxes, classes)
        originals += 1

        for i in range(args.copies):
            result = transform(image=image, bboxes=boxes, class_labels=classes)
            if not result["bboxes"]:
                continue
            name = f"{img_path.stem}_aug{i}"
            cv2.imwrite(
                str(out / "images" / "train" / f"{name}.jpg"),
                result["image"],
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            write_labels(
                out / "labels" / "train" / f"{name}.txt",
                result["bboxes"],
                result["class_labels"],
            )
            created += 1

    val_count = copy_split(source, out, "val")

    (ROOT / "data_aug.yaml").write_text(
        "path: ./dataset_aug\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 4\n"
        "names: ['game', 'upgrade', 'advertising', 'bonus']\n",
        encoding="utf-8",
    )
    print(f"Оригиналов train: {originals}, аугментированных копий: {created}")
    print(f"Валидация без изменений: {val_count}")
    print(f"Датасет: {out}")
    print(f"Конфиг: {ROOT / 'data_aug.yaml'}")


if __name__ == "__main__":
    main()
