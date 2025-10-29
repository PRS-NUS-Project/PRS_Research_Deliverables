import json
import os
import shutil
from dataclasses import dataclass
from typing import List

from config import DEFAULT_CONFIG, DatasetConfig


@dataclass
class ProcessingStats:
    original_total: int = 0
    after_cleaning_total: int = 0
    after_merge_total: int = 0
    skipped_no_annotations: int = 0
    skipped_missing_frames: int = 0
    skipped_duplicate_frames: int = 0

    @property
    def total_skipped(self) -> int:
        return (
            self.skipped_no_annotations
            + self.skipped_missing_frames
            + self.skipped_duplicate_frames
        )

    def print_summary(self, final_frame_count: int):
        print("\n[SUMMARY]")
        print(f"  Original images total: {self.original_total}")
        print(f"  After cleaning (wanted categories only): {self.after_cleaning_total}")
        print(f"  After merging & deduplication: {self.after_merge_total}")
        print(f"  Skipped (no wanted annotations): {self.skipped_no_annotations}")
        print(f"  Skipped (missing frame files): {self.skipped_missing_frames}")
        print(f"  Skipped (duplicate frame names): {self.skipped_duplicate_frames}")
        print(f"  Total skipped: {self.total_skipped}")
        print(f"  Final copied frames: {final_frame_count}")


@dataclass
class DatasetCleaner:
    config: DatasetConfig
    stats = ProcessingStats()

    def clean_json_file(self, file_name: str) -> None:
        raw_filepath = self.config.paths.raw_json_path(file_name)
        with open(raw_filepath) as f:
            data = json.load(f)

        original_images_count = len(data["images"])
        self.stats.original_total += original_images_count

        data["categories"] = [
            category
            for category in data["categories"]
            if int(category["id"]) in self.config.wanted_categories.keys()
        ]

        used_image_ids = set()
        cleaned_annotations = []
        for annotation in data["annotations"]:
            if int(annotation["category_id"]) in self.config.wanted_categories.keys():
                cleaned_annotations.append(annotation)
                used_image_ids.add(int(annotation["image_id"]))
        data["annotations"] = cleaned_annotations

        data["images"] = [
            image for image in data["images"] if int(image["id"]) in used_image_ids
        ]
        cleaned_images_count = len(data["images"])

        self.stats.skipped_no_annotations += (
            original_images_count - cleaned_images_count
        )
        self.stats.after_cleaning_total += cleaned_images_count

        cleaned_file_path = self.config.paths.cleaned_json_path(file_name)
        os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
        with open(cleaned_file_path, "w") as f:
            json.dump(data, f, indent=4)

    def merge_json_files(self, file_names: List[str], output_file_name: str) -> None:
        merged = {"images": [], "categories": [], "annotations": []}

        used_image_ids = []
        used_annotation_ids = []
        frame_name_to_image_id = {}
        missing_frames_image_ids = []

        for file_name in file_names:
            file_path = self.config.paths.cleaned_json_path(file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

            for category in data["categories"]:
                if (
                    category["name"]
                    != self.config.wanted_categories[int(category["id"])]
                ):
                    raise Exception(
                        f"[ERROR] Category ID mismatch for '{category['name']}'"
                    )

            if not merged["categories"]:
                merged["categories"].extend(data["categories"])

            for image in data["images"]:
                image_id = image["id"]
                frame_name = image["file_name"]

                if image_id in used_image_ids:
                    raise Exception(
                        f"[ERROR] Duplicate image ID detected across files: {image_id}"
                    )

                if not os.path.isfile(self.config.paths.raw_image_path(frame_name)):
                    missing_frames_image_ids.append(image_id)
                    self.stats.skipped_missing_frames += 1
                    print(
                        f"[WARNING] Missing frame '{frame_name}' (image id {image_id}) — skipped."
                    )
                    continue

                if frame_name in frame_name_to_image_id:
                    self.stats.skipped_duplicate_frames += 1
                    continue

                merged["images"].append(image)
                used_image_ids.append(image_id)
                frame_name_to_image_id[frame_name] = image_id

            for annotation in data["annotations"]:
                annotation_id = annotation["id"]

                if annotation_id in used_annotation_ids:
                    raise Exception(
                        f"[ERROR] Duplicate annotation ID detected: {annotation_id}"
                    )

                corresponding_image_id = annotation["image_id"]

                if corresponding_image_id in missing_frames_image_ids:
                    continue

                corresponding_frame_name = None
                for image in data["images"]:
                    if image["id"] == corresponding_image_id:
                        corresponding_frame_name = image["file_name"]
                        break

                if corresponding_frame_name is None:
                    raise Exception(
                        f"[ERROR] Annotation {annotation_id} references missing image id {corresponding_image_id}"
                    )

                mapped_image_id = frame_name_to_image_id.get(corresponding_frame_name)
                if mapped_image_id is not None:
                    annotation["image_id"] = mapped_image_id
                    merged["annotations"].append(annotation)
                    used_annotation_ids.append(annotation_id)

        self.stats.after_merge_total = len(merged["images"])

        output_path = self.config.paths.cleaned_json_path(output_file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=4)

        print(f"[INFO] Total merged images: {len(merged['images'])}")
        print(f"[INFO] Total categories: {len(merged['categories'])}")
        print(f"[INFO] Total annotations: {len(merged['annotations'])}")

    def copy_valid_frames(self, merged_file_name: str) -> None:
        with open(self.config.paths.cleaned_json_path(merged_file_name)) as f:
            data = json.load(f)

        cleaned_images_dir = self.config.paths.cleaned_image_path()

        if os.path.exists(cleaned_images_dir):
            shutil.rmtree(cleaned_images_dir)
        os.makedirs(cleaned_images_dir)

        used_frame_names = set()
        for image in data["images"]:
            frame_name = image["file_name"]

            if frame_name in used_frame_names:
                raise Exception(
                    f"[ERROR] Duplicate frame file name detected across image ids: {frame_name}"
                )

            used_frame_names.add(frame_name)
            shutil.copyfile(
                self.config.paths.raw_image_path(frame_name),
                self.config.paths.cleaned_image_path(frame_name),
            )

    def process_dataset(self) -> None:
        print("[INFO] Cleaning individual JSON files...")
        for file_name in self.config.raw_json_files:
            self.clean_json_file(file_name)

        print("[INFO] Merging cleaned JSON files...")
        self.merge_json_files(self.config.raw_json_files, self.config.merged_json_file)

        print("[INFO] Removing temporary cleaned JSON files...")
        for file_name in self.config.raw_json_files:
            file_path = self.config.paths.cleaned_json_path(file_name)
            os.remove(file_path)

        print("[INFO] Copying valid frames...")
        self.copy_valid_frames(self.config.merged_json_file)

        final_frame_count = len(os.listdir(self.config.paths.cleaned_image_path()))
        self.stats.print_summary(final_frame_count)


def main():
    cleaner = DatasetCleaner(DEFAULT_CONFIG)
    cleaner.process_dataset()


if __name__ == "__main__":
    main()
