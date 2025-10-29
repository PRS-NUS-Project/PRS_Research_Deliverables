import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import DEFAULT_CONFIG, DatasetConfig


class AnnotationVisualizer:
    def __init__(self, config: DatasetConfig, output_dir: str = "annotated_images"):
        self.config = config
        self.output_dir = output_dir
        self.category_colors = self.default_colors()
        self.data = self.load_data()
        self.image_id_to_file = {
            img["id"]: img["file_name"] for img in self.data["images"]
        }
        self.annotations_by_image = self.group_annotations()

        os.makedirs(self.output_dir, exist_ok=True)

    def default_colors(self) -> Dict[int, Tuple[int, int, int]]:
        return {
            6: (255, 0, 0),
            15: (0, 255, 0),
            8: (0, 0, 255),
            14: (0, 255, 255),
            5: (255, 0, 255),
            10: (255, 255, 0),
        }

    def set_category_colors(self, colors: Dict[int, Tuple[int, int, int]]) -> None:
        self.category_colors = colors

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

    def visualize_all(self) -> None:
        for image_id, file_name in self.image_id_to_file.items():
            self.visualize_image(image_id, file_name)

        print(f"[INFO] Annotated images saved to {self.output_dir}")

    def visualize_image(self, image_id: int, file_name: str) -> None:
        src_path = os.path.join(self.config.paths.cleaned_image_path(), file_name)
        img = cv2.imread(src_path)

        if img is None:
            print(f"[WARNING] Could not read image {file_name}, skipping")
            return

        anns = self.annotations_by_image.get(image_id, [])

        img = self.draw_annotations(img, anns)
        img = self.draw_legend(img)

        out_path = os.path.join(self.output_dir, file_name)
        cv2.imwrite(out_path, img)

    def draw_annotations(self, img: np.ndarray, annotations: List[dict]) -> np.ndarray:
        for ann in annotations:
            cat_id = int(ann["category_id"])
            x, y, w, h = map(int, ann["bbox"])
            color = self.category_colors.get(cat_id, (255, 255, 255))

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img,
                self.config.wanted_categories[cat_id],
                (x, max(y - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return img

    def draw_legend(
        self,
        img: np.ndarray,
        legend_width: int = 200,
        row_height: int = 20,
        alpha: float = 0.6,
    ) -> np.ndarray:
        legend_height = row_height * len(self.config.wanted_categories)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (legend_width, legend_height), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        for i, (cat_id, name) in enumerate(self.config.wanted_categories.items()):
            color = self.category_colors.get(int(cat_id), (255, 255, 255))
            y_pos = row_height * i + 15

            cv2.rectangle(img, (5, row_height * i + 5), (20, y_pos), color, -1)
            cv2.putText(
                img,
                name,
                (25, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return img


def main():
    print("=== Visualizing Annotations ===")
    visualizer = AnnotationVisualizer(DEFAULT_CONFIG)
    visualizer.visualize_all()


if __name__ == "__main__":
    main()
