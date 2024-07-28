import random
from typing import List, Tuple

import cv2
import numpy as np


__all__ = [
    "RandomResize", "Resize", "resize_img", "resize_bboxes", "resize_polys"
]


def resize_img(
    img: np.ndarray, 
    new_max_edge: Tuple[int, int], 
) -> np.ndarray:
    """
    Args:
    - `img`: `np.ndarray`, shape `(img_h, img_w, 3)`
    - `new_max_edge`: `Tuple[int, int]`, `(new_max_long_edge, new_max_short_edge)`

    Returns:
    `new_img`: `np.ndarray`, shape `(img_h, img_w, 3)`
    """
    img_h, img_w = img.shape[:2]
    new_max_long_edge, new_max_short_edge = new_max_edge

    if img_h > img_w:
        long_edge = img_h
        short_edge = img_w
    else:
        long_edge = img_w
        short_edge = img_h

    long_edge_scale = new_max_long_edge / long_edge
    short_edge_scale = new_max_short_edge / short_edge
    scale = min(long_edge_scale, short_edge_scale)
    
    new_long_edge = int(scale * long_edge)
    new_short_edge = int(scale * short_edge)

    if img_h > img_w:
        new_h = new_long_edge
        new_w = new_short_edge
    else:
        new_h = new_short_edge
        new_w = new_long_edge

    new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR)

    return new_img


def resize_bboxes(
    bboxes: np.ndarray,
    old_img_size: Tuple[int, int],
    new_img_size: Tuple[int, int],
) -> np.ndarray:
    """
    Args:
    - `bboxes`: `np.ndarray`, shape `(num_bboxes, 4)`, bbox format xywh
    - `old_img_size`: `Tuple[int, int]`, `(old_img_h, old_img_w)`
    - `new_img_size`: `Tuple[int, int]`, `(new_img_h, new_img_w)`

    Returns:
    - `new_bboxes`: `np.ndarray`, shape `(num_bboxes, 4)`, bbox format xywh
    """
    old_h, old_w = old_img_size
    new_h, new_w = new_img_size

    h_scale = new_h / old_h
    w_scale = new_w / old_w

    old_bboxes_ws = bboxes[:, 2] - bboxes[:, 0]
    old_bboxes_hs = bboxes[:, 3] - bboxes[:, 1]

    new_bboxes = bboxes.copy()
    new_bboxes[:, 0] = bboxes[:, 0] * w_scale
    new_bboxes[:, 1] = bboxes[:, 1] * h_scale
    new_bboxes[:, 2] = (bboxes[:, 0] + old_bboxes_ws) * w_scale
    new_bboxes[:, 3] = (bboxes[:, 1] + old_bboxes_hs) * h_scale

    new_bboxes = np.round(new_bboxes).astype(bboxes.dtype)
    
    return new_bboxes


def resize_polys(
    polys: List[np.ndarray],
    old_img_size: Tuple[int, int],
    new_img_size: Tuple[int, int],
) -> List[np.ndarray]:
    """
    Args:
    - `polys`: `List[np.ndarray]`; `(num_polys, (num_points, 2)`
    - `old_img_size`: `Tuple[int, int]`, `(old_img_h, old_img_w)`
    - `new_img_size`: `Tuple[int, int]`, `(new_img_h, new_img_w)`

    Returns:
    - `new_polys`: `List[np.ndarray]`; `(num_polys, (num_points, 2))`
    """
    old_h, old_w = old_img_size
    new_h, new_w = new_img_size

    h_scale = new_h / old_h
    w_scale = new_w / old_w

    new_polys = []

    for poly in polys:
        new_poly = poly.copy()
        new_poly[:, 0] = poly[:, 0] * w_scale
        new_poly[:, 1] = poly[:, 1] * h_scale
        new_polys.append(new_poly)
    
    return new_polys
    

class RandomResize:
    def __init__(self, new_max_edges: List[Tuple[int, int]]) -> None:
        """
        Args:
        - `new_max_edges`: `List[(long_edge, short_edge)]`
        """
        self.new_max_edges = new_max_edges

    def __call__(
        self, 
        img: np.ndarray,
        bboxes: np.ndarray,
        polys: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
        - `img`: `np.ndarray`, `(img_h, img_w, 3)`
        - `bboxes`: `np.ndarray`, `(num_objs, 4)`, bbox format xywh
        - `polys`: `np.ndarray`, `(num_objs, (num_points, 2))`

        Returns:
        - `new_img`: `np.ndarray`, `(img_h, img_w, 3)`
        - `new_bboxes`: `np.ndarray`, `(num_objs, box_dim)`
        - `new_polys`: `np.ndarray`, `(num_objs, (num_points, 2))`
        """
        new_max_edge = random.choice(self.new_max_edges)
        old_img_size = img.shape[:2]

        new_img = resize_img(img, new_max_edge)
        new_img_size = new_img[:2]
        
        new_bboxes = resize_bboxes(bboxes, old_img_size, new_img_size)
        new_polys = resize_polys(polys, old_img_size, new_img_size)

        return new_img, new_bboxes, new_polys
    
class Resize:
    def __init__(self, new_max_edge: Tuple[int, int]) -> None:
        """
        Args:
        - `new_max_edge`: `(long_edge, short_edge)`
        """
        self.new_max_edge = new_max_edge

    def __call__(
        self, 
        img: np.ndarray,
        bboxes: np.ndarray,
        polys: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
        - `img`: `np.ndarray`, `(img_h, img_w, 3)`
        - `bboxes`: `np.ndarray`, `(num_objs, 4)`, bbox format xywh
        - `polys`: `List[np.ndarray]`, `(num_objs, (num_points, 2))`

        Returns:
        - `new_img`: `np.ndarray`, `(img_h, img_w, 3)`
        - `new_bboxes`: `np.ndarray`, `(num_objs, box_dim)`
        - `new_polys`: `np.ndarray`, `(num_objs, (num_points, 2))`
        """
        old_img_size = img.shape[:2]

        new_img = resize_img(img, self.new_max_edge)
        new_img_size = new_img[:2]
        
        new_bboxes = resize_bboxes(bboxes, old_img_size, new_img_size)
        new_polys = resize_polys(polys, old_img_size, new_img_size)

        return new_img, new_bboxes, new_polys