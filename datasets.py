import os
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(
        self,
        ann_p: Union[str, os.PathLike],
        img_root: Union[str, os.PathLike]
    ) -> None:
        super(CocoDataset, self).__init__()
        self.ann_p = ann_p
        self.img_root = img_root

        self._make_datasets()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, img_idx: int) -> Dict[str, Any]:
        img_data = self.data[img_idx]
        return img_data
    
    def _make_datasets(self) -> None:
        self.data: List[Dict[str, Any]]  = []
        self.category_name_id_dict: Dict[str, int] = {}
        self.category_id_name_dict: Dict[int, str] = {}

        coco = COCO(self.ann_p)

        # Create category name-id dicts
        for cat_idx, cat_info in coco.cats.items():
            cat_id = int(cat_info["id"])
            cat_name = cat_info["name"]

            self.category_id_name_dict[cat_id] = cat_name
            self.category_name_id_dict[cat_name] = cat_id

        # Iterate over the img list,
        # add img_p and the bbox and mask annotations
        # of the img to self.data.
        for img_id, img_info in coco.imgs.items():
            img_name = Path(img_info['file_name']).name
            img_p = os.path.join(self.img_root, img_name)

            # axes (num_anns, )
            categories: List[int] = []

            # axes (num_anns, box_dim)
            bboxes: List[Tuple[int, int, int, int]] = []

            # axes (num_anns, (num_polys, (num_poly_points * 2, )))
            polygons: List[Tuple[int, ...]] = []

            # Retrieve annotations of this img
            img_ann_ids = coco.getAnnIds(img_id)
            img_anns = [coco.anns[ann_id] for ann_id in img_ann_ids]

            for ann in img_anns:
                category_id = int(ann["category_id"])
                bbox = ann["bbox"]
                polygon = ann["segmentation"]

                categories.append(category_id)
                bboxes.append(bbox)
                polygons.append(polygon)

            categories = np.asarray(categories, dtype = np.int32)
            bboxes = np.asarray(bboxes, dtype = np.int32)
            
            img_data = {
                "img_p": img_p,
                "categories": categories,
                "bboxes": bboxes,
                "polygons": polygons
            }

            self.data.append(img_data)