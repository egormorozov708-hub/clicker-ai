"""Clicker AI — бот для игры «Кликер: Становление шкафа».

Что умеет:
  * находит основной кликабельный объект (класс game) и кликает по нему;
  * раз в 30 секунд покупает одно улучшение по очереди (класс improvement),
    затем сразу возвращается к кликам по game;
  * закрывает рекламу крестиком (класс advertising);
  * нажимает бонусные кнопки 2X / +1K (класс bonus);
  * показывает превью с рамками (--preview) или работает без окна.

Инференс — ONNX Runtime, без зависимости от Ultralytics в рабочем боте.

Запуск:
    python app.py --model best.onnx
    python app.py --model best.onnx --preview --dry-run
    python app.py --model best.onnx --window "Кликер" --crop-window

Горячие клавиши (если установлен pynput):
    Пробел — остановка
    F8     — пауза / продолжить
    F9     — выход
    Esc    — выход
Без pynput: Ctrl+C в консоли.

Важно: бот кликает только когда активно окно игры (проверка по заголовку).
Реклама закрывается автоматически; отключить: --no-close-ads.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyautogui
import pygetwindow as gw

try:
    import mss
except ImportError:  # mss не обязателен: есть резервный pyautogui.screenshot
    mss = None

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

# На Windows при масштабировании интерфейса (125 %, 150 %) координаты pyautogui
# могут не совпадать с пикселями скриншота. Включаем DPI-awareness.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

COLORS = {
    "game": (0, 255, 60),
    "improvement": (255, 200, 0),
    "advertising": (255, 0, 255),
    "bonus": (0, 255, 255),
}

# В старых весах класс назывался upgrade — поддерживаем оба имени.
ALIASES = {"upgrade": "improvement"}
CLASS_NAMES_DEFAULT = ["game", "improvement", "advertising", "bonus"]


def load_class_names() -> list[str]:
    """Имена классов из classes.txt рядом с app.py, exe или в _MEIPASS."""
    candidates = [Path(__file__).with_name("classes.txt")]
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).with_name("classes.txt"))
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "classes.txt")
    for path in candidates:
        if path.exists():
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            names = [name for name in names if name]
            if names:
                return names
    return CLASS_NAMES_DEFAULT


@dataclass
class Config:
    model: str = "best.onnx"
    window: str = "Кликер"
    crop_window: bool = False
    preview: bool = False
    dry_run: bool = False
    imgsz: int = 640
    conf: float = 0.45
    iou: float = 0.45
    max_det: int = 100
    threads: int = 0
    click_interval: float = 0.01
    click_burst: int = 3
    native_click: bool = False
    upgrade_interval: float = 30.0
    bonus_interval: float = 12.0
    ad_cooldown: float = 3.0
    ad_settle: float = 1.0
    improvement_settle: float = 0.4
    empty_retry: float = 1.0
    window_refresh: float = 1.0
    verbose: bool = False
    click_game: bool = True
    click_improvements: bool = True
    click_bonuses: bool = True
    close_ads: bool = True
    allow_any_window: bool = False
    max_seconds: float = 0.0


@dataclass
class Stats:
    frames: int = 0
    game_clicks: int = 0
    improvement_clicks: int = 0
    bonus_clicks: int = 0
    ad_clicks: int = 0
    started: float = field(default_factory=time.time)
    last_game_click: float = 0.0
    last_ad_click: float = 0.0
    resume_game_at: float = 0.0
    next_upgrade_at: float = 0.0
    next_bonus_at: float = 0.0
    # Стабильные ID кнопок, уже нажатых в текущем круге. Индексация по
    # пересортированному списку ломалась, когда UI менял состав кнопок.
    improvement_seen: list[tuple[float, float]] = field(default_factory=list)
    bonus_seen: list[tuple[float, float]] = field(default_factory=list)

    @property
    def clicks(self) -> int:
        return self.game_clicks + self.improvement_clicks + self.bonus_clicks + self.ad_clicks

    @property
    def fps(self) -> float:
        elapsed = max(1e-6, time.time() - self.started)
        return self.frames / elapsed


def find_window(title_part: str):
    candidates = []
    for window in gw.getAllWindows():
        try:
            title = (window.title or "").strip()
            if not title or title_part.lower() not in title.lower():
                continue
            if window.width <= 300 or window.height <= 200:
                continue
            if getattr(window, "isMinimized", False):
                continue
            candidates.append(window)
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda w: w.width * w.height)


def list_windows() -> None:
    print("Окна с заголовками:")
    for window in gw.getAllWindows():
        title = (window.title or "").strip()
        if title and window.width > 200:
            print(f"  {title!r}  ({window.left},{window.top},{window.width},{window.height})")


_mss_instance = None
_mss_disabled = False


def grab_screen(region: tuple[int, int, int, int] | None) -> np.ndarray:
    """Захват экрана через mss (быстро), с резервом на pyautogui."""
    global _mss_instance, _mss_disabled
    if mss is not None and not _mss_disabled:
        try:
            if _mss_instance is None:
                factory = getattr(mss, "MSS", None) or mss.mss
                _mss_instance = factory()
            monitor = region if region is not None else _mss_instance.monitors[1]
            shot = _mss_instance.grab(monitor)
            # mss отдаёт BGRA. cvtColor быстрее ломтика с copy() и сразу
            # делает массив непрерывным — детектору так тоже быстрее.
            return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            _mss_disabled = True
            print(f"mss недоступен ({exc}); перехожу на pyautogui.screenshot.")
    image = pyautogui.screenshot(region=region)
    frame = np.array(image)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def center(box: list[float]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def stable_id(box: list[float], frame_w: int, frame_h: int) -> tuple[float, float]:
    """Нормализованный центр кнопки — не зависит от порядка сортировки."""
    cx, cy = center(box)
    return cx / max(1, frame_w), cy / max(1, frame_h)


def already_seen(
    point: tuple[float, float], seen: list[tuple[float, float]]
) -> bool:
    """Совпадение кнопки с уже нажатой с допуском 2 % кадра.

    Допуск убирает ложные «новые кнопки» при дрожании детекции на несколько
    пикселей и при небольшом сдвиге окна.
    """
    return any(
        abs(point[0] - other[0]) <= 0.02 and abs(point[1] - other[1]) <= 0.02
        for other in seen
    )


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def game_box_reason(box: list[float], frame_w: int, frame_h: int) -> str | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 35:
        return "слишком маленькая рамка"
    if h / max(w, 1.0) < 0.65:
        return "не вытянутая форма"
    if (x1 + x2) / 2 > frame_w * 0.97:  # правая колонка с рекламой
        return "правая колонка с рекламой"
    return None


def improvement_box_reason(box: list[float], frame_w: int, frame_h: int) -> str | None:
    """Кнопки покупки — широкие, в нижней левой части игрового поля."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 70 or h < 28:
        return "слишком маленькая рамка"
    if not (1.0 <= w / max(h, 1.0) <= 8.0):
        return "неверные пропорции"
    if x1 > frame_w * 0.65 or y1 < frame_h * 0.45:
        return "не нижняя левая часть"
    return None


def bonus_box_reason(box: list[float], frame_w: int, frame_h: int) -> str | None:
    """2X / +1K — квадратные кнопки в верхней левой части."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 55 or h < 45:
        return "слишком маленькая рамка"
    if not (0.5 <= w / max(h, 1.0) <= 2.2):
        return "неверные пропорции"
    if x2 > frame_w * 0.75 or y2 > frame_h * 0.45:
        return "не верхняя левая часть"
    return None


def ad_box_reason(box: list[float], frame_w: int, frame_h: int) -> str | None:
    """Крестик рекламы: маленький, почти квадратный, в правой части кадра.

    Попап может располагаться выше или ниже в зависимости от положения окна
    браузера, поэтому по вертикали ограничение мягкое. Фильтр не даёт спутать
    крестик с широкой кнопкой «Отключить рекламу» и с мелкими кнопками окна.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if not (40 <= w <= 200 and 40 <= h <= 200):
        return "неподходящий размер"
    if not (0.6 <= w / max(h, 1.0) <= 2.2):
        return "неверные пропорции"
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if cx < frame_w * 0.45 or cy > frame_h * 0.85:
        return "не правый верх"
    return None


def valid_game_box(box: list[float], frame_w: int, frame_h: int) -> bool:
    return game_box_reason(box, frame_w, frame_h) is None


def valid_improvement_box(box: list[float], frame_w: int, frame_h: int) -> bool:
    return improvement_box_reason(box, frame_w, frame_h) is None


def valid_bonus_box(box: list[float], frame_w: int, frame_h: int) -> bool:
    return bonus_box_reason(box, frame_w, frame_h) is None


def valid_ad_box(box: list[float], frame_w: int, frame_h: int) -> bool:
    return ad_box_reason(box, frame_w, frame_h) is None


BOX_REASONS = {
    "game": game_box_reason,
    "improvement": improvement_box_reason,
    "advertising": ad_box_reason,
    "bonus": bonus_box_reason,
}


def rejection_details(
    detections: list[Detection], frame_w: int, frame_h: int
) -> dict[str, Counter]:
    """Причины, по которым рамки не прошли геометрические фильтры (--verbose)."""
    details: dict[str, Counter] = {}
    for name, _score, box in detections:
        checker = BOX_REASONS.get(name)
        reason = checker(box, frame_w, frame_h) if checker else "неизвестный класс"
        if reason:
            details.setdefault(name, Counter())[reason] += 1
    return details


Detection = tuple[str, float, list[float]]


class OnnxDetector:
    """YOLOv8-детектор на ONNX Runtime: letterbox, инференс и NMS вручную."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.45,
        iou: float = 0.45,
        max_det: int = 100,
        imgsz: int = 640,
        threads: int = 0,
    ):
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.names = [canonical(name) for name in load_class_names()]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # На 12-поточном CPU авторежим ONNX Runtime проигрывает явным 8
        # потокам из-за SMT. Ограничиваем авторежим восемью потоками;
        # --threads по-прежнему задаёт значение вручную.
        if threads > 0:
            options.intra_op_num_threads = int(threads)
        else:
            options.intra_op_num_threads = min(8, os.cpu_count() or 1)
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        shape = self.session.get_inputs()[0].shape
        model_size = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else None
        if model_size:
            if int(imgsz) != int(model_size):
                print(
                    f"Размер входа ONNX — {model_size}x{model_size}; "
                    f"параметр --imgsz {imgsz} игнорируется."
                )
            self.imgsz = int(model_size)
        else:
            self.imgsz = int(imgsz)

    @staticmethod
    def _letterbox(image: np.ndarray, size: int):
        height, width = image.shape[:2]
        ratio = min(size / height, size / width)
        new_w, new_h = round(width * ratio), round(height * ratio)
        if (new_w, new_h) != (width, height):
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = size - new_w, size - new_h
        left, top = pad_w // 2, pad_h // 2
        image = cv2.copyMakeBorder(
            image, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return image, ratio, (left, top)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            union = area_i + area_rest - inter
            overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = rest[overlap <= iou_thres]
        return np.asarray(keep, dtype=np.int64)

    def detect(self, frame: np.ndarray, conf: float | None = None) -> list[Detection]:
        conf = self.conf if conf is None else float(conf)
        image, ratio, (pad_x, pad_y) = self._letterbox(frame, self.imgsz)
        # Один вызов вместо cvtColor + transpose + astype + деления:
        # blobFromImage делает то же самое в C++ и экономит ~2–3 мс на кадр.
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, swapRB=True, crop=False
        )

        output = self.session.run(self.output_names, {self.input_name: blob})[0]
        pred = np.asarray(output, dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            raise RuntimeError(f"Неожиданная форма выхода ONNX: {output.shape}")
        expected = 4 + len(self.names)
        if pred.shape[0] == expected and pred.shape[1] != expected:
            pred = pred.T
        if pred.shape[1] != expected:
            raise RuntimeError(
                f"ONNX вернул {pred.shape[1]} колонок, ожидалось {expected} "
                f"(4 координаты + {len(self.names)} классов)."
            )

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = np.isfinite(scores) & (scores >= conf)
        if not mask.any():
            return []
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes = np.empty_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        frame_h, frame_w = frame.shape[:2]
        kept: list[int] = []
        for cls in np.unique(class_ids):
            idx = np.flatnonzero(class_ids == cls)
            kept.extend(idx[self._nms(boxes[idx], scores[idx], self.iou)].tolist())
        kept.sort(key=lambda i: scores[i], reverse=True)

        detections: list[Detection] = []
        for i in kept[: self.max_det]:
            x1 = (boxes[i, 0] - pad_x) / ratio
            y1 = (boxes[i, 1] - pad_y) / ratio
            x2 = (boxes[i, 2] - pad_x) / ratio
            y2 = (boxes[i, 3] - pad_y) / ratio
            x1, x2 = np.clip([x1, x2], 0, frame_w)
            y1, y2 = np.clip([y1, y2], 0, frame_h)
            name = self.names[int(class_ids[i])] if int(class_ids[i]) < len(self.names) else "object"
            detections.append((name, float(scores[i]), [float(x1), float(y1), float(x2), float(y2)]))
        return detections


def pick(
    detections: list[Detection],
    name: str,
    frame_w: int,
    frame_h: int,
    validator,
) -> Detection | None:
    best = None
    best_score = -1.0
    for detection in detections:
        if detection[0] != name:
            continue
        if not validator(detection[2], frame_w, frame_h):
            continue
        if detection[1] > best_score:
            best = detection
            best_score = detection[1]
    return best


def failsafe_check() -> None:
    """Проверка угла экрана для нативного клика (pyautogui делает её сам)."""
    if not pyautogui.FAILSAFE:
        return
    x, y = pyautogui.position()
    width, height = pyautogui.size()
    if (x, y) in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        raise pyautogui.FailSafeException


def native_click(x: int, y: int) -> bool:
    """Быстрый клик через WinAPI. False, если платформа не поддерживается."""
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.SetCursorPos(int(x), int(y))
        user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
        user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
        return True
    except Exception:
        return False


def click_box(
    box: list[float],
    offset: tuple[int, int],
    cfg: Config,
    label: str,
) -> None:
    """Единый низкоуровневый клик: и для game, и для рекламы, и для кнопок."""
    x, y = center(box)
    if cfg.dry_run:
        if label != "game":
            print(f"[dry-run] клик: {label} в ({offset[0] + x}, {offset[1] + y})")
        return
    px, py = offset[0] + x, offset[1] + y
    if cfg.native_click:
        failsafe_check()
        if native_click(px, py):
            return
    pyautogui.click(px, py)


def click_game(
    detections: list[Detection],
    frame_w: int,
    frame_h: int,
    offset: tuple[int, int],
    cfg: Config,
    stats: Stats,
    now: float,
) -> bool:
    best = pick(detections, "game", frame_w, frame_h, valid_game_box)
    if best is None:
        return False
    if now - stats.last_game_click < cfg.click_interval:
        return False
    # За один кадр делаем несколько кликов: инференс ограничивает цикл
    # примерно 15–16 FPS, поэтому без серии быстрее кликать не получится.
    for index in range(cfg.click_burst):
        if index:
            time.sleep(cfg.click_interval)
        click_box(best[2], offset, cfg, "game")
        stats.game_clicks += 1
    stats.last_game_click = time.time()
    return True


def close_ad(
    detections: list[Detection],
    frame_w: int,
    frame_h: int,
    offset: tuple[int, int],
    cfg: Config,
    stats: Stats,
    now: float,
) -> bool:
    """Отдельное действие «закрыть рекламу» на той же основе, что и клик game."""
    if not cfg.close_ads:
        return False
    if now - stats.last_ad_click < cfg.ad_cooldown:
        return False
    ad = pick(detections, "advertising", frame_w, frame_h, valid_ad_box)
    if ad is None:
        return False
    click_box(ad[2], offset, cfg, "advertising")
    stats.ad_clicks += 1
    stats.last_ad_click = now
    stats.resume_game_at = now + cfg.ad_settle  # даём окну рекламы закрыться
    prefix = "[dry-run] " if cfg.dry_run else ""
    print(f"{prefix}Реклама закрыта (уверенность {ad[1]:.2f}), возвращаюсь к game.")
    return True


def buy_improvement(
    detections: list[Detection],
    frame_w: int,
    frame_h: int,
    offset: tuple[int, int],
    cfg: Config,
    stats: Stats,
    now: float,
) -> bool:
    """Раз в upgrade_interval — одно улучшение, следующее по очереди."""
    if not cfg.click_improvements:
        return False
    if now < stats.next_upgrade_at:
        return False
    improvements = sorted(
        (
            d for d in detections
            if d[0] == "improvement"
            and valid_improvement_box(d[2], frame_w, frame_h)
        ),
        key=lambda d: (d[2][1], d[2][0]),
    )
    if not improvements:
        stats.next_upgrade_at = now + cfg.empty_retry
        return False
    # Стабильные ID вместо индекса в пересортированном списке: новая кнопка
    # или сдвиг старой больше не сбивает очередь. Когда все видимые кнопки
    # нажаты, начинается новый круг.
    buttons = [
        (position, stable_id(d[2], frame_w, frame_h), d)
        for position, d in enumerate(improvements)
    ]
    fresh = [
        item for item in buttons
        if not already_seen(item[1], stats.improvement_seen)
    ]
    if not fresh:
        stats.improvement_seen.clear()
        fresh = buttons
    position, button_id, detection = fresh[0]
    stats.improvement_seen.append(button_id)
    click_box(
        detection[2], offset, cfg,
        f"improvement {position + 1}/{len(improvements)}",
    )
    stats.improvement_clicks += 1
    stats.next_upgrade_at = now + cfg.upgrade_interval
    stats.resume_game_at = now + cfg.improvement_settle
    prefix = "[dry-run] " if cfg.dry_run else ""
    print(
        f"{prefix}Купил улучшение {position + 1}/{len(improvements)}; "
        f"следующее через {cfg.upgrade_interval:.0f} с, возвращаюсь к game."
    )
    return True


def click_bonus(
    detections: list[Detection],
    frame_w: int,
    frame_h: int,
    offset: tuple[int, int],
    cfg: Config,
    stats: Stats,
    now: float,
) -> bool:
    """2X / +1K: нажимаем по очереди, если бонус виден."""
    if not cfg.click_bonuses:
        return False
    if now < stats.next_bonus_at:
        return False
    bonuses = sorted(
        (
            d for d in detections
            if d[0] == "bonus" and valid_bonus_box(d[2], frame_w, frame_h)
        ),
        key=lambda d: (d[2][1], d[2][0]),
    )
    if not bonuses:
        stats.next_bonus_at = now + cfg.empty_retry
        return False
    buttons = [
        (position, stable_id(d[2], frame_w, frame_h), d)
        for position, d in enumerate(bonuses)
    ]
    fresh = [
        item for item in buttons
        if not already_seen(item[1], stats.bonus_seen)
    ]
    if not fresh:
        stats.bonus_seen.clear()
        fresh = buttons
    position, button_id, detection = fresh[0]
    stats.bonus_seen.append(button_id)
    click_box(
        detection[2], offset, cfg,
        f"bonus {position + 1}/{len(bonuses)}",
    )
    stats.bonus_clicks += 1
    stats.next_bonus_at = now + cfg.bonus_interval
    return True


def draw(
    frame: np.ndarray,
    detections: list[Detection],
    stats: Stats,
    paused: bool,
) -> np.ndarray:
    canvas = frame.copy()
    for name, score, box in detections:
        color = COLORS.get(name, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, f"{name} {score:.2f}", (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    status = "PAUSED" if paused else "RUNNING"
    text = (
        f"{status} | FPS {stats.fps:.1f} | game {stats.game_clicks} | "
        f"improvement {stats.improvement_clicks} | ad {stats.ad_clicks}"
    )
    cv2.putText(canvas, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


class Preview:
    """Окно превью: OpenCV, а при сборке без GUI — Tkinter."""

    def __init__(self, max_width: int = 1280, max_height: int = 720):
        self.max_width = max_width
        self.max_height = max_height
        self.backend = ""
        self.closed = False
        self._tk = None
        self._label = None
        self._photo = None
        try:
            cv2.namedWindow("Clicker AI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Clicker AI", max_width, max_height)
            self.backend = "opencv"
            return
        except cv2.error:
            pass

        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Не удалось открыть окно превью: OpenCV собран без GUI, "
                "а Tkinter/Pillow недоступны. Запустите без --preview."
            ) from exc

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.title("Clicker AI")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.bind("<Escape>", lambda _event: self.close())
        self._root.bind("<space>", lambda _event: self.close())
        self._root.bind("q", lambda _event: self.close())
        self._label = tk.Label(self._root, bg="black")
        self._label.pack(fill="both", expand=True)
        self.backend = "tkinter"
        print("Превью: OpenCV без GUI, использую окно Tkinter.")

    def show(self, frame: np.ndarray) -> int:
        """Показать кадр. Возвращает код клавиши как cv2.waitKey (255 = нет)."""
        if self.closed:
            return -1
        if self.backend == "opencv":
            cv2.imshow("Clicker AI", frame)
            key = cv2.waitKey(1) & 0xFF
            try:
                visible = cv2.getWindowProperty("Clicker AI", cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = 0
            if visible < 1:
                self.closed = True
                return -1
            return key

        height, width = frame.shape[:2]
        scale = min(self.max_width / width, self.max_height / height, 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(rgb)
        self._photo = self._ImageTk.PhotoImage(image)
        self._label.configure(image=self._photo)
        self._root.update_idletasks()
        self._root.update()
        return -1 if self.closed else 255

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.backend == "tkinter":
            try:
                self._root.destroy()
            except Exception:
                pass
        elif self.backend == "opencv":
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def setup_hotkeys(state: dict):
    try:
        from pynput import keyboard
    except Exception:
        return None

    def on_press(key):
        if key == keyboard.Key.space:
            print("Пробел — остановка.")
            state["running"] = False
            return False
        if key == keyboard.Key.f8:
            state["paused"] = not state["paused"]
            print("Пауза" if state["paused"] else "Продолжаем")
        elif key in (keyboard.Key.f9, keyboard.Key.esc):
            state["running"] = False
            return False
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clicker AI bot (ONNX Runtime)")
    parser.add_argument("--model", default="best.onnx", help="путь к ONNX-модели")
    parser.add_argument("--window", default="Кликер", help="часть заголовка окна игры")
    parser.add_argument("--crop-window", action="store_true", help="захватывать только окно игры")
    parser.add_argument("--preview", action="store_true", help="показывать окно с детекцией")
    parser.add_argument("--dry-run", action="store_true", help="только детекция, без кликов")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="размер входа; для ONNX берётся из модели")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="порог уверенности детекций, 0..1")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="порог IoU для NMS, 0..1")
    parser.add_argument("--max-det", type=int, default=100,
                        help="максимум детекций на кадр")
    parser.add_argument("--threads", type=int, default=0,
                        help="потоки ONNX Runtime; 0 = авто (до 8)")
    parser.add_argument("--click-interval", type=float, default=0.01,
                        help="пауза между кликами по game, секунды (по умолчанию 0.01)")
    parser.add_argument("--click-burst", type=int, default=3,
                        help="кликов по game за один кадр (1 = по одному клику)")
    parser.add_argument("--native-click", action="store_true",
                        help="быстрый клик через WinAPI вместо pyautogui")
    parser.add_argument("--upgrade-interval", type=float, default=30.0,
                        help="как часто покупать одно улучшение, секунды")
    parser.add_argument("--bonus-interval", type=float, default=12.0,
                        help="как часто нажимать бонус 2X/+1K, секунды")
    parser.add_argument("--ad-cooldown", type=float, default=3.0,
                        help="пауза между попытками закрыть рекламу, секунды")
    parser.add_argument("--ad-settle", type=float, default=1.0,
                        help="пауза после закрытия рекламы, секунды")
    parser.add_argument("--improvement-settle", type=float, default=0.4,
                        help="пауза после покупки улучшения, секунды")
    parser.add_argument("--empty-retry", type=float, default=1.0,
                        help="повтор, если нужные кнопки не найдены, секунды")
    parser.add_argument("--window-refresh", type=float, default=1.0,
                        help="как часто обновлять геометрию окна, секунды (0 = каждый кадр)")
    parser.add_argument("--verbose", action="store_true",
                        help="показывать причины отбраковки детекций")
    parser.add_argument("--no-game", action="store_true")
    parser.add_argument("--no-improvements", "--no-upgrades", dest="no_improvements",
                        action="store_true")
    parser.add_argument("--no-bonuses", action="store_true")
    parser.add_argument("--no-close-ads", action="store_true",
                        help="не закрывать рекламу автоматически")
    parser.add_argument("--close-ads", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-any-window", action="store_true",
                        help="кликать без найденного окна игры (по всему экрану)")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 = без ограничения")
    parser.add_argument("--list-windows", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("conf", "iou"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} должен быть в диапазоне 0..1")
    if args.max_det < 1:
        parser.error("--max-det должен быть >= 1")
    if args.threads < 0:
        parser.error("--threads должен быть >= 0")
    for name in (
        "click_interval", "upgrade_interval", "bonus_interval", "ad_cooldown",
        "ad_settle", "improvement_settle", "empty_retry", "window_refresh",
        "max_seconds",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} должен быть >= 0")
    if args.click_burst < 1:
        parser.error("--click-burst должен быть >= 1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_windows:
        list_windows()
        return
    validate_args(parser, args)

    cfg = Config(
        model=args.model,
        window=args.window,
        crop_window=args.crop_window,
        preview=args.preview,
        dry_run=args.dry_run,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        threads=args.threads,
        click_interval=args.click_interval,
        click_burst=args.click_burst,
        native_click=args.native_click,
        upgrade_interval=args.upgrade_interval,
        bonus_interval=args.bonus_interval,
        ad_cooldown=args.ad_cooldown,
        ad_settle=args.ad_settle,
        improvement_settle=args.improvement_settle,
        empty_retry=args.empty_retry,
        window_refresh=args.window_refresh,
        verbose=args.verbose,
        click_game=not args.no_game,
        click_improvements=not args.no_improvements,
        click_bonuses=not args.no_bonuses,
        close_ads=not args.no_close_ads,
        allow_any_window=args.allow_any_window,
        max_seconds=args.max_seconds,
    )

    model_path = Path(cfg.model)
    if not model_path.exists():
        raise SystemExit(
            f"Модель не найдена: {model_path}. Укажите --model best.onnx "
            "или выполните python ptToOnnx.py."
        )

    print(f"Модель: {model_path}")
    try:
        detector = OnnxDetector(
            model_path,
            conf=cfg.conf,
            iou=cfg.iou,
            max_det=cfg.max_det,
            imgsz=cfg.imgsz,
            threads=cfg.threads,
        )
    except Exception as exc:
        raise SystemExit(f"Не удалось загрузить ONNX-модель: {exc}") from exc
    print(f"ONNX Runtime {ort.__version__}, вход {detector.imgsz}x{detector.imgsz}")
    print("Классы:", {i: name for i, name in enumerate(detector.names)})
    print(
        f"Захват экрана: {'mss' if mss is not None else 'pyautogui'}; "
        f"клик: {'WinAPI' if cfg.native_click else 'pyautogui'}."
    )

    window = find_window(cfg.window)
    if window:
        print(f"Окно игры: {window.title!r} ({window.left},{window.top},{window.width},{window.height})")
    else:
        print(f"Окно «{cfg.window}» не найдено.")
        print("Подсказка: python app.py --list-windows")
        if cfg.allow_any_window:
            print("Включён --allow-any-window: работаю по всему экрану.")
        elif not cfg.dry_run:
            print(
                "Клики отключены, чтобы не нажимать в чужие окна. "
                "Укажите --window или --allow-any-window."
            )

    state = {"running": True, "paused": False}
    listener = setup_hotkeys(state)
    stats = Stats()
    now = stats.started
    stats.next_upgrade_at = now + cfg.upgrade_interval
    stats.next_bonus_at = now + cfg.bonus_interval
    last_window_geometry = (
        (window.left, window.top, window.width, window.height) if window else None
    )
    next_window_check = time.time() + cfg.window_refresh

    print("Старт. Пробел — стоп, F8 — пауза, F9/Esc — выход." if listener else "Старт. Остановка: Ctrl+C.")
    if cfg.dry_run:
        print("Режим --dry-run: кликов не будет, действия только пишутся в лог.")
    print(
        f"Клики по game: {cfg.click_burst} за кадр, пауза "
        f"{cfg.click_interval * 1000:.0f} мс."
    )
    print(
        f"Улучшения: раз в {cfg.upgrade_interval:.0f} с по очереди; "
        f"реклама: {'автоматически' if cfg.close_ads else 'выключена'}."
    )

    preview = None
    if cfg.preview:
        try:
            preview = Preview()
            print(
                "Превью включено для отладки: окно превью перехватывает фокус, "
                "поэтому для реальных кликов запускайте бота без --preview."
            )
        except RuntimeError as exc:
            print(f"Превью недоступно: {exc}")
            cfg.preview = False

    last_log = time.time()
    try:
        while state["running"]:
            if cfg.max_seconds and time.time() - stats.started > cfg.max_seconds:
                break
            if state["paused"]:
                time.sleep(0.1)
                continue

            # Геометрия окна могла измениться: пользователь передвинул окно,
            # изменил размер или перезапустил игру. Обновляем её по таймеру.
            now = time.time()
            if now >= next_window_check:
                refreshed = find_window(cfg.window)
                if refreshed is None:
                    if window is not None:
                        window = None
                        last_window_geometry = None
                        print(f"Окно «{cfg.window}» потеряно; ищу его снова.")
                else:
                    geometry = (
                        refreshed.left, refreshed.top,
                        refreshed.width, refreshed.height,
                    )
                    changed = window is None or geometry != last_window_geometry
                    window = refreshed
                    last_window_geometry = geometry
                    if changed:
                        print(f"Окно игры: {window.title!r} {geometry}")
                next_window_check = now + cfg.window_refresh

            region = None
            offset = (0, 0)
            try:
                if cfg.crop_window and window:
                    region = (window.left, window.top, window.width, window.height)
                    offset = (window.left, window.top)
                frame = grab_screen(region)
            except Exception as exc:
                if region is not None:
                    print(f"Не удалось снять окно ({exc}), снимаю весь экран.")
                    region = None
                    offset = (0, 0)
                    frame = grab_screen(None)
                else:
                    raise

            frame_h, frame_w = frame.shape[:2]
            try:
                detections = detector.detect(frame, cfg.conf)
            except Exception as exc:
                print(f"Ошибка инференса: {exc}. Бот остановлен.")
                break
            detections.sort(key=lambda d: d[1], reverse=True)
            stats.frames += 1

            active_title = ""
            try:
                active = gw.getActiveWindow()
                active_title = (active.title or "") if active else ""
            except Exception:
                pass
            if window is None:
                game_active = cfg.allow_any_window or cfg.dry_run
            else:
                game_active = bool(active_title and cfg.window.lower() in active_title.lower())
            can_click = game_active and not cfg.dry_run

            now = time.time()
            if can_click or cfg.dry_run:
                # 1) Реклама — highest priority. Пока крестик виден, по игре не кликаем,
                # даже если кулдаун ещё не прошёл: попап перекрывает сцену.
                ad_visible = pick(
                    detections, "advertising", frame_w, frame_h, valid_ad_box
                ) is not None
                if ad_visible:
                    close_ad(detections, frame_w, frame_h, offset, cfg, stats, now)
                # 2) Не кликаем по game, пока закрывается реклама/после покупки.
                elif now < stats.resume_game_at:
                    pass
                # 3) Раз в 30 секунд — одно улучшение, затем снова game.
                elif buy_improvement(detections, frame_w, frame_h, offset, cfg, stats, now):
                    pass
                else:
                    if cfg.click_game:
                        click_game(detections, frame_w, frame_h, offset, cfg, stats, now)
                    click_bonus(detections, frame_w, frame_h, offset, cfg, stats, now)

            if cfg.preview and preview is not None:
                shown = draw(frame, detections, stats, state["paused"])
                key = preview.show(shown)
                if key == -1 or key in (ord("q"), 27, 32):
                    break

            if now - last_log >= 1.0:
                names = ", ".join(
                    f"{n}:{sum(1 for d in detections if d[0] == n)}"
                    for n in ("game", "improvement", "advertising", "bonus")
                )
                until_upgrade = max(0, int(stats.next_upgrade_at - now))
                print(
                    f"FPS {stats.fps:.1f} | game {stats.game_clicks} | "
                    f"improvement {stats.improvement_clicks} | bonus {stats.bonus_clicks} | "
                    f"ad {stats.ad_clicks} | до улучшения {until_upgrade} с | объекты: {names}"
                    + ("" if game_active else " | окно игры не активно")
                    + (" [dry-run]" if cfg.dry_run else "")
                )
                if cfg.verbose:
                    rejected = rejection_details(detections, frame_w, frame_h)
                    for name, reasons in rejected.items():
                        details = ", ".join(
                            f"{reason} x{count}"
                            for reason, count in reasons.most_common()
                        )
                        print(f"  отброшено {name}: {details}")
                last_log = now

    except pyautogui.FailSafeException:
        print("\nСработала защита pyautogui (мышь в углу). Остановлено.")
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        if preview is not None:
            preview.close()
        if listener:
            listener.stop()
        print(
            f"Итог: кадров {stats.frames}, кликов {stats.clicks} "
            f"(game {stats.game_clicks}, improvement {stats.improvement_clicks}, "
            f"bonus {stats.bonus_clicks}, ad {stats.ad_clicks}), "
            f"время {time.time() - stats.started:.1f} c."
        )


if __name__ == "__main__":
    main()
