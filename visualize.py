from typing import Tuple, List

import cv2
import numpy as np


__all__ = ["draw_bboxes", "draw_polys"]


def draw_bboxes(
    img: np.ndarray,
    bboxes: np.ndarray,
    bbox_color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Args:
    - `img`: `np.ndarray`, `(img_h, img_w, 3)`
    - `bboxes`: `np.ndarray`, `(num_bboxes, 4)`, xywh
    - `bbox_color`: `Tuple[int, int, int]`, BGR color code

    Returns:
    - `img_with_bboxes`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    thickness = 2

    for bbox in bboxes:
        x1, y1 = bbox[:2]
        w, h = bbox[2:]
        x2 = x1 + w
        y2 = y1 + h
        img = cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness)

    return img

def draw_polys(
    img: np.ndarray,
    polys: List[np.ndarray],
    poly_color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Args:
    - `img`: `np.ndarray`, `(img_h, img_w, 3)`
    - `polys`: `List[np.ndarray]`, `(num_objs, (num_points, 2))`, 
    - `poly_color`: `Tuple[int, int, int]`, BGR color code

    Returns:
    - `img_with_polys`: `np.ndarray`, `(img_h, img_w, 3)`
    """
    thickness = 2

    for poly in polys:
        poly = poly.reshape((-1, 1, 2))
        img = cv2.polylines(img, [poly], True, poly_color, thickness)

    return img