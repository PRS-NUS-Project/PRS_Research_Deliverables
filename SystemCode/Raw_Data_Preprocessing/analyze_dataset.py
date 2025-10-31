import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DatasetConfig, PathConfig


@dataclass
class AnnotationStats:
    total_good: int = 0
    total_bad: int = 0

    @property
    def total(self) -> int:
        return self.total_good + self.total_bad

    def add_counts(self, good: int, bad: int):
        self.total_good += good
        self.total_bad += bad

    def print_summary(self, title: str = "ANNOTATION SUMMARY"):
        print(f"\n{'='*60}")
        print(f"{title:^60}")
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
    image_ids: Set[Tuple[str, int]] = field(default_factory=set)
    instances: int = 0

    def copy(self):
        return CategoryInfo(
            name=self.name, image_ids=self.image_ids.copy(), instances=self.instances
        )


class CategoryCounter:
    def __init__(self, wanted_categories: Dict[int, str], config_id: str = ""):
        self.categories = {
            cid: CategoryInfo(name=name) for cid, name in wanted_categories.items()
        }
        self.config_id = config_id

    def add_counts_from_file(self, file_path: str, wanted_categories: Dict[int, str]):
        with open(file_path, "r") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            cid = ann["category_id"]
            if cid in self.categories:
                self.categories[cid].instances += 1
                self.categories[cid].image_ids.add((self.config_id, ann["image_id"]))

    def copy(self):
        new_counter = CategoryCounter({}, self.config_id)
        new_counter.categories = {
            cid: cat.copy() for cid, cat in self.categories.items()
        }
        return new_counter

    def merge(self, other: "CategoryCounter"):
        for cid, other_cat in other.categories.items():
            if cid in self.categories:
                self.categories[cid].instances += other_cat.instances
                self.categories[cid].image_ids.update(other_cat.image_ids)

    def print_summary(self, title: str = "CATEGORY BREAKDOWN"):
        all_image_ids = set()
        total_instances = 0
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print(f"{'='*70}")
        print(f"  {'Category':<30} {'Images':<15} {'Instances':<15}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")
        for cid, cat in sorted(self.categories.items()):
            img_count = len(cat.image_ids)
            inst_count = cat.instances
            print(f"  {cat.name:<30} {img_count:<15,} {inst_count:<15,}")
            all_image_ids.update(cat.image_ids)
            total_instances += inst_count
        print(f"  {'-'*30} {'-'*15} {'-'*15}")
        print(f"  {'TOTAL':<30} {len(all_image_ids):<15,} {total_instances:<15,}")
        print(f"{'='*70}\n")

    def plot_categories(self, output_path: str = "category_analysis.png"):
        if not self.categories:
            print("[WARNING] No categories to plot")
            return
        sorted_categories = sorted(self.categories.items())
        names = [cat.name for _, cat in sorted_categories]
        img_counts = [len(cat.image_ids) for _, cat in sorted_categories]
        inst_counts = [cat.instances for _, cat in sorted_categories]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Category Distribution Analysis", fontsize=16, fontweight="bold")
        x = range(len(names))
        cmap = matplotlib.colormaps["Set3"]
        colors = cmap(range(len(names)))
        ax1.bar(x, img_counts, color=colors, edgecolor="black", linewidth=1.2)
        ax1.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Images", fontsize=12, fontweight="bold")
        ax1.set_title("Images per Category", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        for i, v in enumerate(img_counts):
            ax1.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
        ax2.bar(x, inst_counts, color=colors, edgecolor="black", linewidth=1.2)
        ax2.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Number of Instances", fontsize=12, fontweight="bold")
        ax2.set_title("Instances per Category", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        for i, v in enumerate(inst_counts):
            ax2.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Category analysis plot saved to {output_path}")


def analyze_multiple_configs(configs: List[DatasetConfig]):
    combined_stats = AnnotationStats()
    wanted_categories = configs[0].wanted_categories
    combined_category_counter = CategoryCounter(wanted_categories, "combined")

    per_config_counters = []

    for idx, cfg in enumerate(configs):
        file_path = cfg.paths.cleaned_json_path(cfg.merged_json_file)
        dataset_name = (
            file_path.split("/")[-2] if "/" in file_path else f"Config {idx+1}"
        )

        print(f"\n{'#'*70}")
        print(f"  Analyzing dataset: {dataset_name}")
        print(f"  Path: {file_path}")
        print(f"{'#'*70}")

        with open(file_path, "r") as f:
            data = json.load(f)

        if "categories" in data:
            file_cats = {
                c["id"]: c["name"]
                for c in data["categories"]
                if c["id"] in wanted_categories
            }
            for cid, cname in wanted_categories.items():
                assert (
                    cid in file_cats
                ), f"[ERROR] Missing category ID {cid} in {file_path}"
                found_name = file_cats[cid]
                assert found_name == cname, (
                    f"[ERROR] Name mismatch for category {cid} in {file_path}\n"
                    f"Expected: '{cname}', Found: '{found_name}'"
                )

        anns = data.get("annotations", [])
        good = sum(1 for a in anns if a["category_id"] in wanted_categories)
        bad = len(anns) - good

        config_stats = AnnotationStats()
        config_stats.add_counts(good, bad)
        config_stats.print_summary(f"{dataset_name.upper()} ANNOTATION SUMMARY")

        config_category_counter = CategoryCounter(wanted_categories, dataset_name)
        config_category_counter.add_counts_from_file(file_path, wanted_categories)
        config_category_counter.print_summary(
            f"{dataset_name.upper()} CATEGORY BREAKDOWN"
        )

        per_config_counters.append(config_category_counter)

        combined_stats.add_counts(good, bad)
        combined_category_counter.merge(config_category_counter)

    print(f"\n\n{'='*70}")
    print(f"{'FINAL COMBINED RESULTS':^70}")
    print(f"{'='*70}")

    combined_stats.print_summary("FINAL COMBINED ANNOTATION SUMMARY")
    combined_category_counter.print_summary("FINAL COMBINED CATEGORY BREAKDOWN")

    print(f"\n{'='*70}")
    print(f"{'VALIDATION':^70}")
    print(f"{'='*70}")

    for cid, combined_cat in combined_category_counter.categories.items():
        expected_instances = sum(
            counter.categories[cid].instances for counter in per_config_counters
        )
        assert combined_cat.instances == expected_instances, (
            f"[ERROR] Instance count mismatch for category {combined_cat.name}\n"
            f"Expected: {expected_instances}, Got: {combined_cat.instances}"
        )
        print(
            f"  ✓ Category '{combined_cat.name}': {combined_cat.instances:,} instances (verified)"
        )

    print(f"\n  Per-Category Image Count Validation:")
    for cid, combined_cat in sorted(combined_category_counter.categories.items()):
        per_config_image_counts = [
            len(counter.categories[cid].image_ids) for counter in per_config_counters
        ]
        expected_sum = sum(per_config_image_counts)
        actual_combined = len(combined_cat.image_ids)

        status = "✓" if actual_combined == expected_sum else "✗"
        print(f"    {status} {combined_cat.name}:")
        print(
            f"       Expected: {expected_sum:,} (sum: {' + '.join(map(lambda x: f'{x:,}', per_config_image_counts))})"
        )
        print(f"       Actual:   {actual_combined:,}")

        if actual_combined != expected_sum:
            overlap = expected_sum - actual_combined
            print(f"       ⚠ OVERLAP: {overlap:,} duplicate image IDs detected!")

    all_image_ids_per_config = [
        set().union(*(cat.image_ids for cat in counter.categories.values()))
        for counter in per_config_counters
    ]

    total_images_in_combined = len(
        set().union(
            *(cat.image_ids for cat in combined_category_counter.categories.values())
        )
    )
    sum_of_images = sum(len(img_ids) for img_ids in all_image_ids_per_config)

    print(f"\n  Overall Image ID Statistics:")
    print(
        f"    Total unique (config, image_id) pairs in combined: {total_images_in_combined:,}"
    )
    print(f"    Sum of unique images per config: {sum_of_images:,}")

    if total_images_in_combined == sum_of_images:
        print(f"  ✓ No overlap - each image ID is unique per config")
    else:
        overlap = sum_of_images - total_images_in_combined
        print(f"  ✗ WARNING: {overlap:,} image IDs appear in multiple configs!")

    print(f"\n  ✓ All accumulation validations passed!")
    print(f"{'='*70}\n")

    combined_category_counter.plot_categories()


def main():
    wanted_categories = {
        1: "Inferior Epigastric Vessels",
        2: "Pubic Bone",
        3: "Testicular Vessels",
        4: "Triangle of Doom",
        5: "Triangle of Pain",
        6: "Vas Deferens",
    }
    configs = [
        DatasetConfig(
            wanted_categories=wanted_categories,
            merged_json_file="_annotations.coco.json",
            paths=PathConfig(cleaned_json_dir="640_tep_dataset/train"),
        ),
        DatasetConfig(
            wanted_categories=wanted_categories,
            merged_json_file="_annotations.coco.json",
            paths=PathConfig(cleaned_json_dir="640_tep_dataset/test"),
        ),
        DatasetConfig(
            wanted_categories=wanted_categories,
            merged_json_file="_annotations.coco.json",
            paths=PathConfig(cleaned_json_dir="640_tep_dataset/valid"),
        ),
    ]
    analyze_multiple_configs(configs)


if __name__ == "__main__":
    main()
