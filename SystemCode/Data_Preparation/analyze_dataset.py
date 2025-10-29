import json
from dataclasses import dataclass, field
from typing import Dict, Set

from config import DEFAULT_CONFIG, DatasetConfig


@dataclass
class AnnotationStats:
    total_good: int = 0
    total_bad: int = 0

    @property
    def total(self) -> int:
        return self.total_good + self.total_bad

    def print_summary(self):
        print(f"Total annotations count: {self.total}")
        if self.total > 0:
            bad_percentage = self.total_bad / self.total * 100
            good_percentage = self.total_good / self.total * 100
            print(
                f"Total annotations that will not be used: {self.total_bad} "
                f"which is {bad_percentage:.2f}% of annotations"
            )
            print(
                f"Total annotations that will be used: {self.total_good} "
                f"which is {good_percentage:.2f}% of annotations"
            )


@dataclass
class CategoryInfo:
    name: str
    image_ids: Set[int] = field(default_factory=set)
    instances: int = 0


class AnnotationCounter:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.stats = AnnotationStats()

    def count_annotations_in_file(self, file_path: str) -> tuple[int, int]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])

        good_annotations = [
            annotation
            for annotation in annotations
            if annotation.get("category_id") in self.config.wanted_categories.keys()
        ]

        found_count = len(annotations)
        good_count = len(good_annotations)
        bad_count = found_count - good_count

        print(f"File: {file_path}")
        print(f"  Annotations found: {found_count}")
        print(f"  Annotations that will not be used: {bad_count}")
        print(f"  Annotations that will be used: {good_count}")
        print()

        return good_count, bad_count

    def analyze_merged_file(self, file_name: str) -> None:
        file_path = self.config.paths.cleaned_json_path(file_name)
        good_count, bad_count = self.count_annotations_in_file(file_path)

        self.stats.total_good += good_count
        self.stats.total_bad += bad_count

        self.stats.print_summary()


@dataclass
class CategoryCounter:
    config: DatasetConfig
    categories: Dict[int, CategoryInfo] = field(init=False)

    def __post_init__(self):
        self.categories = {}
        for cat_id, cat_name in self.config.wanted_categories.items():
            self.categories[cat_id] = CategoryInfo(name=cat_name)

    def count_categories_in_file(self, file_path: str) -> None:
        with open(file_path) as f:
            data = json.load(f)

        for annotation in data["annotations"]:
            cat_id = annotation["category_id"]
            if cat_id in self.categories:
                self.categories[cat_id].instances += 1
                self.categories[cat_id].image_ids.add(annotation["image_id"])

    def print_summary(self) -> None:
        all_image_ids = set()
        total_instances = 0

        for cat_id, category in self.categories.items():
            print(f"Category {cat_id} which is '{category.name}':")
            print(f"    Associated Images = {len(category.image_ids)}")
            print(f"    Total Instances = {category.instances}")

            all_image_ids.update(category.image_ids)
            total_instances += category.instances

        print()
        print(f"Total unique images: {len(all_image_ids)}")
        print(f"Total instances: {total_instances}")

    def analyze_merged_file(self, file_name: str) -> None:
        file_path = self.config.paths.cleaned_json_path(file_name)
        self.count_categories_in_file(file_path)
        self.print_summary()


def count_annotations():
    counter = AnnotationCounter(DEFAULT_CONFIG)
    counter.analyze_merged_file(DEFAULT_CONFIG.merged_output_filename)


def count_categories():
    counter = CategoryCounter(DEFAULT_CONFIG)
    counter.analyze_merged_file(DEFAULT_CONFIG.merged_output_filename)


def main():
    print("=== Annotation Analysis ===")
    count_annotations()
    print("\n" + "=" * 50 + "\n")
    print("=== Category Analysis ===")
    count_categories()


if __name__ == "__main__":
    main()
