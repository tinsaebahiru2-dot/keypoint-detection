import argparse
import json
import math
import typing
from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
import torch
from torchvision.transforms import ToTensor
import albumentations as A

from keypoint_detection.data.coco_parser import (
    CocoImage,
    CocoKeypointCategory,
    CocoKeypoints,
)
from keypoint_detection.data.imageloader import ImageDataset, ImageLoader
from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.types import COCO_KEYPOINT_TYPE, IMG_KEYPOINTS_TYPE


class COCOTopDownKeypointsDataset(ImageDataset):
    """
    Top-down dataset that returns a crop per person bbox with the corresponding keypoints remapped
    into the crop coordinate system. The format returned matches the base dataset:
    (image_tensor, List[channel][List[[u,v]]]) where each channel holds 0/1 keypoints for that crop.
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("COCOTopDownKeypointsDataset")
        parser.add_argument("--crop_width", type=int, default=256)
        parser.add_argument("--crop_height", type=int, default=256)
        parser.add_argument(
            "--bbox_scale", type=float, default=1.25, help="Scale factor around bbox for context"
        )
        parser.add_argument(
            "--detect_only_visible_keypoints",
            dest="detect_only_visible_keypoints",
            default=True,
            action="store_true",
            help="If set, only keypoints with flag > 1.0 will be used.",
        )
        parser.add_argument(
            "--image_root_dir",
            type=str,
            help="Optional root directory for images. If provided, images are loaded from this dir joined with file_name from JSON.",
        )
        return parent_parser

    def __init__(
        self,
        json_dataset_path: str,
        keypoint_channel_configuration: list[list[str]],
        crop_width: int = 256,
        crop_height: int = 256,
        bbox_scale: float = 1.25,
        detect_only_visible_keypoints: bool = True,
        transform: Optional[Any] = None,
        imageloader: Optional[ImageLoader] = None,
        **kwargs,
    ):
        super().__init__(imageloader if imageloader is not None else ImageLoader())
        self.image_to_tensor_transform = ToTensor()
        self.dataset_json_path = Path(json_dataset_path)
        self.dataset_dir_path = self.dataset_json_path.parent
        image_root_dir = kwargs.get("image_root_dir", None)
        self.image_root_dir_path = Path(image_root_dir) if image_root_dir else self.dataset_dir_path
        self.keypoint_channel_configuration = keypoint_channel_configuration
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.bbox_scale = bbox_scale
        self.detect_only_visible_keypoints = detect_only_visible_keypoints
        self.transform = transform
        self.samples = self._prepare_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Tuple[torch.Tensor, IMG_KEYPOINTS_TYPE]:
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        rel_image_path, crop_xywh, channel_keypoints = self.samples[index]
        image_path = self.image_root_dir_path / rel_image_path
        image = self.image_loader.get_image(str(image_path), index)
        if image.shape[2] == 4:
            image = image[..., :3]

        x, y, w, h = crop_xywh
        x0 = max(0, int(math.floor(x)))
        y0 = max(0, int(math.floor(y)))
        x1 = min(image.shape[1], int(math.ceil(x + w)))
        y1 = min(image.shape[0], int(math.ceil(y + h)))

        crop_np = image[y0:y1, x0:x1, :]
        if crop_np.size == 0:
            crop = np.zeros((self.crop_height, self.crop_width, 3), dtype=image.dtype)
        else:
            resized = A.Resize(self.crop_height, self.crop_width)(image=crop_np)
            crop = resized["image"]

        keypoints = channel_keypoints
        if self.transform:
            transformed = self.transform(image=crop, keypoints=keypoints)
            crop, keypoints = transformed["image"], transformed["keypoints"]

        # Make sure we return integer pixel coords (as expected by heatmap gen)
        keypoints = [
            [[int(math.floor(kp[0])), int(math.floor(kp[1]))] for kp in ch]
            for ch in keypoints
        ]

        # help type checker understand attribute type
        image_loader: ImageLoader = self.image_loader
        crop_tensor = self.image_to_tensor_transform(crop)
        return crop_tensor, keypoints

    def _prepare_samples(self):  # noqa: C901
        with open(self.dataset_json_path, "r") as file:
            data = json.load(file)
            parsed_coco = CocoKeypoints(**data)

        img_dict: typing.Dict[int, CocoImage] = {}
        for img in parsed_coco.images:
            img_dict[img.id] = img

        category_dict: typing.Dict[int, CocoKeypointCategory] = {}
        for category in parsed_coco.categories:
            category_dict[category.id] = category

        samples = []
        for ann in parsed_coco.annotations:
            if ann.bbox is None:
                continue
            img_meta = img_dict[ann.image_id]
            cat = category_dict[ann.category_id]
            all_names = cat.keypoints

            # convert flat list to [[u,v,flag], ...]
            kps = COCOKeypointsDataset.split_list_in_keypoints(ann.keypoints)

            # compute scaled bbox square-ish crop
            bx, by, bw, bh = ann.bbox
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            side = max(bw, bh) * self.bbox_scale
            bx = cx - side / 2.0
            by = cy - side / 2.0
            bw = side
            bh = side

            # remap keypoints into crop and keep visibility filter
            channel_keypoints: List[List[List[float]]] = [[] for _ in range(len(self.keypoint_channel_configuration))]
            for ch_idx, ch_names in enumerate(self.keypoint_channel_configuration):
                for name in ch_names:
                    try:
                        idx = all_names.index(name)
                    except ValueError:
                        continue
                    u, v, f = kps[idx]
                    if self.detect_only_visible_keypoints:
                        if f <= 1.5:
                            continue
                    else:
                        if f <= 0.5:
                            continue
                    # inside image check
                    if u < 0 or v < 0 or u > img_meta.width or v > img_meta.height:
                        continue
                    u_rel = (u - bx) / max(1e-6, bw) * self.crop_width
                    v_rel = (v - by) / max(1e-6, bh) * self.crop_height
                    if 0 <= u_rel < self.crop_width and 0 <= v_rel < self.crop_height:
                        channel_keypoints[ch_idx].append([u_rel, v_rel])

            samples.append([img_meta.file_name, [bx, by, bw, bh], channel_keypoints])
        return samples

    @staticmethod
    def collate_fn(data):
        return COCOKeypointsDataset.collate_fn(data)


