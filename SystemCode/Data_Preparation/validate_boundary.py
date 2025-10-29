import json
import os
from typing import Dict, List

import cv2

from config import DEFAULT_CONFIG, DatasetConfig


class BoundingBoxValidator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = self.load_data()
        self.image_id_to_file = {
            img["id"]: img["file_name"] for img in self.data["images"]
        }
        self.annotations_by_image = self.group_annotations()

    def load_data(self) -> dict:
        merged_path = self.config.paths.cleaned_json_path(
            self.config.merged_output_filename
        )
        with open(merged_path) as f:
            return json.load(f)

    def group_annotations(self) -> Dict[int, List[dict]]:
        annotations_by_image = {}
        for ann in self.data["annotations"]:
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)
        return annotations_by_image

    def validate_all(self, frame_prefix: str = "frame_") -> None:
        for image_id, file_name in self.image_id_to_file.items():
            if not file_name.startswith(frame_prefix):
                continue

            self.validate_image(image_id, file_name)

    def validate_image(self, image_id: int, file_name: str) -> None:
        img_path = os.path.join(self.config.paths.cleaned_image_path(), file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] Could not read image {file_name}")
            return

        h, w, _ = img.shape
        anns = self.annotations_by_image.get(image_id, [])

        for ann in anns:
            self.validate_bbox(ann["bbox"], file_name, w, h)

    def validate_bbox(
        self, bbox: List[float], file_name: str, img_w: int, img_h: int
    ) -> None:
        x, y, bw, bh = bbox
        x2, y2 = x + bw, y + bh

        if bw <= 0 or bh <= 0:
            print(f"[ZERO SIZE] {file_name}, bbox: {bbox}")

        if x < 0 or y < 0 or x2 > img_w or y2 > img_h:
            print(
                f"[OUT OF BOUNDS] {file_name}, bbox: {bbox}, img size: ({img_w}, {img_h})"
            )


def main():
    print("=== Validating Bounding Boxes ===")
    validator = BoundingBoxValidator(DEFAULT_CONFIG)
    validator.validate_all()


if __name__ == "__main__":
    main()
