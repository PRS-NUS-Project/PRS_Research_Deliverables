import json
from dataclasses import dataclass, field
from typing import Dict, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
from config import DEFAULT_CONFIG, DatasetConfig


@dataclass
class AnnotationStats:
    total_good: int = 0
    total_bad: int = 0

    @property
    def total(self) -> int:
        return self.total_good + self.total_bad

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"{'ANNOTATION SUMMARY':^60}")
        print(f"{'='*60}")
        print(f"  Total Annotations:        {self.total:>10,}")

        if self.total > 0:
            bad_percentage = self.total_bad / self.total * 100
            good_percentage = self.total_good / self.total * 100

            print(
                f"  ├─ Used:                  {self.total_good:>10,}  ({good_percentage:>5.1f}%)"
            )
            print(
                f"  └─ Excluded:              {self.total_bad:>10,}  ({bad_percentage:>5.1f}%)"
            )
        print(f"{'='*60}\n")


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

        print(f"\n  File: {file_path}")
        print(f"    Total Found:     {found_count:>6,}")
        print(f"    ├─ Used:         {good_count:>6,}")
        print(f"    └─ Excluded:     {bad_count:>6,}")

        return good_count, bad_count

    def analyze_merged_file(self) -> None:
        file_path = self.config.paths.cleaned_json_path(self.config.merged_json_file)
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

        print(f"\n{'='*70}")
        print(f"{'CATEGORY BREAKDOWN':^70}")
        print(f"{'='*70}")
        print(f"  {'Category':<30} {'Images':<15} {'Instances':<15}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")

        for cat_id, category in sorted(self.categories.items()):
            images_count = len(category.image_ids)
            instances_count = category.instances

            print(f"  {category.name:<30} {images_count:<15,} {instances_count:<15,}")

            all_image_ids.update(category.image_ids)
            total_instances += category.instances

        print(f"  {'-'*30} {'-'*15} {'-'*15}")
        print(f"  {'TOTAL':<30} {len(all_image_ids):<15,} {total_instances:<15,}")
        print(f"{'='*70}\n")

    def plot_categories(self, output_path: str = "category_analysis.png") -> None:
        if not self.categories:
            print("[WARNING] No categories to plot")
            return

        sorted_categories = sorted(self.categories.items())
        category_names = [cat.name for _, cat in sorted_categories]
        images_counts = [len(cat.image_ids) for _, cat in sorted_categories]
        instances_counts = [cat.instances for _, cat in sorted_categories]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Category Distribution Analysis", fontsize=16, fontweight="bold")

        x_pos = range(len(category_names))
        cmap = matplotlib.colormaps['Set3']
        colors = cmap(range(len(category_names)))

        ax1.bar(x_pos, images_counts, color=colors, edgecolor="black", linewidth=1.2)
        ax1.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Images", fontsize=12, fontweight="bold")
        ax1.set_title("Images per Category", fontsize=14, fontweight="bold")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(category_names, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        for i, v in enumerate(images_counts):
            ax1.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")

        ax2.bar(x_pos, instances_counts, color=colors, edgecolor="black", linewidth=1.2)
        ax2.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Number of Instances", fontsize=12, fontweight="bold")
        ax2.set_title("Instances per Category", fontsize=14, fontweight="bold")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(category_names, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        for i, v in enumerate(instances_counts):
            ax2.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Category analysis plot saved to {output_path}")

    def analyze_merged_file(self) -> None:
        file_path = self.config.paths.cleaned_json_path(self.config.merged_json_file)
        self.count_categories_in_file(file_path)
        self.print_summary()
        self.plot_categories()


def count_annotations():
    counter = AnnotationCounter(DEFAULT_CONFIG)
    counter.analyze_merged_file()


def count_categories():
    counter = CategoryCounter(DEFAULT_CONFIG)
    counter.analyze_merged_file()


def main():
    count_annotations()
    count_categories()


if __name__ == "__main__":
    main()
