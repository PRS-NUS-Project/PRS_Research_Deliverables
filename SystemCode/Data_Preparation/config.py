import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PathConfig:
    raw_images_dir: str = os.path.join("raw_dataset", "Image Files")
    raw_json_dir: str = os.path.join("raw_dataset", "JSON Files")
    cleaned_images_dir: str = os.path.join("cleaned_dataset", "Image Files")
    cleaned_json_dir: str = os.path.join("cleaned_dataset", "JSON Files")

    def raw_image_path(self, file_name: str = "") -> str:
        return os.path.join(self.raw_images_dir, file_name)

    def cleaned_image_path(self, file_name: str = "") -> str:
        return os.path.join(self.cleaned_images_dir, file_name)

    def raw_json_path(self, file_name: str = "") -> str:
        return os.path.join(self.raw_json_dir, file_name)

    def cleaned_json_path(self, file_name: str = "") -> str:
        return os.path.join(self.cleaned_json_dir, file_name)


@dataclass
class DatasetConfig:
    wanted_categories: Dict[int, str]
    json_files: list[str]
    merged_output_filename: str = "coco-merged.json"
    paths: PathConfig = field(default_factory=PathConfig)


DEFAULT_CONFIG = DatasetConfig(
    wanted_categories={
        5: "Triangle of Doom",
        6: "Triangle of Pain",
        8: "Vas Deferens",
        10: "Pubic Bone",
        14: "Inferior Epigastric Vessels",
        15: "Testicular Vessels",
    },
    json_files=[
        "coco-1756384709.315322.json",
        "coco-1756384724.5534823.json",
        "coco-1756384753.9625523.json",
    ],
)
