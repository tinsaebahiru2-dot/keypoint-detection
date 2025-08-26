import math
from typing import List

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from keypoint_detection.utils.visualization import draw_keypoints_on_image


def _resize_and_pad(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    target_h, target_w = size_hw
    pil = Image.fromarray(img)
    pil = pil.resize((target_w, target_h))
    return np.array(pil)


def run_topdown_inference(
    model: KeypointDetector,
    image: Image.Image,
    yolo_model: YOLO,
    person_conf: float = 0.25,
) -> Image.Image:
    orig_w, orig_h = image.size
    np_img = np.array(image)
    yolo_res = yolo_model.predict(source=np_img, classes=[0], conf=person_conf, verbose=False)[0]  # 0=person

    keypoints_all: List[List[List[int]]] = [[[] for _ in model.keypoint_channel_configuration] for _ in range(1)]
    for box in yolo_res.boxes.xyxy.cpu().numpy().tolist():
        x0, y0, x1, y1 = box
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        cx = x0 + w / 2.0
        cy = y0 + h / 2.0
        side = max(w, h) * 1.25
        bx = cx - side / 2.0
        by = cy - side / 2.0
        bw = side
        bh = side

        # crop and resize to model input size
        x0i = max(0, int(math.floor(bx)))
        y0i = max(0, int(math.floor(by)))
        x1i = min(orig_w, int(math.ceil(bx + bw)))
        y1i = min(orig_h, int(math.ceil(by + bh)))
        crop = np_img[y0i:y1i, x0i:x1i, :]
        if crop.size == 0:
            continue
        resized = _resize_and_pad(crop, (np_img.shape[0], np_img.shape[1]))
        tensored_image = torch.from_numpy(resized).float() / 255.0
        tensored_image = tensored_image.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            heatmaps = model(tensored_image)
        keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=0.1)
        image_keypoints = keypoints[0]

        # map back to original image coords
        for ch_idx, ch_kps in enumerate(image_keypoints):
            for kp in ch_kps:
                u_rel, v_rel = kp
                u = u_rel / resized.shape[1] * bw + bx
                v = v_rel / resized.shape[0] * bh + by
                keypoints_all[0][ch_idx].append([int(u), int(v)])

    out = draw_keypoints_on_image(image, keypoints_all[0], model.keypoint_channel_configuration)
    return out


if __name__ == "__main__":
    wandb_checkpoint = ""
    image_path = ""
    yolo = YOLO("yolov8s.pt")

    image = Image.open(image_path).convert("RGB")
    model = get_model_from_wandb_checkpoint(wandb_checkpoint)
    out = run_topdown_inference(model, image, yolo)
    out.save("inference_result_topdown.png")



