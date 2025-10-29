import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DatasetConfig


@dataclass
class BBoxMetrics:
    area_ratios: List[float] = field(default_factory=list)
    width_ratios: List[float] = field(default_factory=list)
    height_ratios: List[float] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)


class BoundingBoxAnalyzer:

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = self.load_data()
        self.image_id_to_file = {
            img["id"]: img["file_name"] for img in self.data["images"]
        }
        self.annotations_by_image = self.group_annotations()
        self.metrics = BBoxMetrics()

    def load_data(self) -> dict:
        merged_path = self.config.paths.cleaned_json_path(self.config.merged_json_file)
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

    def analyze_all(self, frame_prefix: str = "frame_") -> None:
        for image_id, file_name in self.image_id_to_file.items():
            if not file_name.startswith(frame_prefix):
                continue
            self.analyze_image(image_id, file_name)

    def analyze_image(self, image_id: int, file_name: str) -> None:
        img_path = os.path.join(self.config.paths.cleaned_image_path(), file_name)
        img = cv2.imread(img_path)
        if img is None:
            return

        h, w, _ = img.shape
        img_area = w * h
        anns = self.annotations_by_image.get(image_id, [])

        for ann in anns:
            x, y, bw, bh = ann["bbox"]

            if bw <= 0 or bh <= 0:
                continue

            bbox_area = bw * bh
            area_ratio = bbox_area / img_area
            width_ratio = bw / w
            height_ratio = bh / h
            aspect_ratio = bw / bh

            self.metrics.area_ratios.append(area_ratio)
            self.metrics.width_ratios.append(width_ratio)
            self.metrics.height_ratios.append(height_ratio)
            self.metrics.aspect_ratios.append(aspect_ratio)

    def plot_metrics(
        self, output_path: str = "bbox_analysis.png", bins: int = 50
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Bounding Box Analysis", fontsize=16, fontweight="bold")

        axes[0, 0].hist(
            self.metrics.area_ratios, bins=bins, color="steelblue", edgecolor="black"
        )
        axes[0, 0].set_xlabel("Area Ratio (BBox Area / Image Area)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("BBox Area Distribution")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(
            self.metrics.width_ratios, bins=bins, color="forestgreen", edgecolor="black"
        )
        axes[0, 1].set_xlabel("Width Ratio (BBox Width / Image Width)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("BBox Width Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(
            self.metrics.height_ratios, bins=bins, color="coral", edgecolor="black"
        )
        axes[1, 0].set_xlabel("Height Ratio (BBox Height / Image Height)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("BBox Height Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(
            self.metrics.aspect_ratios,
            bins=bins,
            color="mediumpurple",
            edgecolor="black",
        )
        axes[1, 1].set_xlabel("Aspect Ratio (BBox Width / BBox Height)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("BBox Aspect Ratio Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Analysis plot saved to {output_path}")
        print(f"[INFO] Total bounding boxes analyzed: {len(self.metrics.area_ratios)}")


class AnnotationVisualizer:

    def __init__(self, config: DatasetConfig, output_dir: str = "annotated_images"):
        self.config = config
        self.output_dir = output_dir
        self.category_colors = self._default_colors()
        self.data = self.load_data()
        self.image_id_to_file = {
            img["id"]: img["file_name"] for img in self.data["images"]
        }
        self.annotations_by_image = self.group_annotations()

        os.makedirs(self.output_dir, exist_ok=True)

    def _default_colors(self) -> Dict[int, Tuple[int, int, int]]:
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
        merged_path = self.config.paths.cleaned_json_path(self.config.merged_json_file)
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


def validate(config: DatasetConfig):
    analyzer = BoundingBoxAnalyzer(config)
    analyzer.validate_all()


def analyze(config: DatasetConfig):
    analyzer = BoundingBoxAnalyzer(config)
    analyzer.analyze_all()
    analyzer.plot_metrics()


def visualize(config: DatasetConfig):
    visualizer = AnnotationVisualizer(config)
    visualizer.visualize_all()


def main():
    config = DatasetConfig()

    print("=== Validating Bounding Boxes ===")
    validate(config)
    print("\n" + "=" * 50 + "\n")
    print("=== Analyzing Bounding Boxes ===")
    analyze(config)
    print("\n" + "=" * 50 + "\n")
    print("=== Visualizing Annotations ===")
    visualize(config)


if __name__ == "__main__":
    main()
