#!/usr/bin/env python3
"""
COCO metrics → plots + CSVs

Creates:
  outdir/image_coverage_union.csv               # image-level (union of masks)
  outdir/annotation_metrics.csv                 # annotation-level metrics
  outdir/hist_*.png                             # the same plots as before

Requires:
  numpy, matplotlib
  pycocotools for true union coverage (falls back to sum if missing)
  pillow if your JSON lacks image width/height and you need to read from disk
"""

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    from pycocotools import mask as maskUtils
    PYCOCO_OK = True
except Exception:
    PYCOCO_OK = False


# ---------- utils ----------

def find_json_files(path: str) -> List[str]:
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return [path]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
        if files:
            return files
    raise FileNotFoundError(f"No JSON files found at: {path}")


def load_image_size(images_dir: str, file_name: str) -> Tuple[int, int]:
    if not PIL_OK:
        raise RuntimeError("JSON lacks image dims and Pillow is not available to read sizes from disk.")
    p = os.path.join(images_dir, file_name)
    with Image.open(p) as im:
        w, h = im.size
    return int(w), int(h)


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d not in (0, 0.0, None) else 0.0


def polygon_area(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    if len(x) < 3:
        return 0.0
    x2 = np.append(x, x[0])
    y2 = np.append(y, y[0])
    return float(abs(np.dot(x2[:-1], y2[1:]) - np.dot(y2[:-1], x2[1:])) * 0.5)


def seg_area_from_ann(ann: Dict[str, Any]) -> float:
    # Prefer COCO-provided "area"
    a = ann.get("area")
    if a is not None:
        try:
            return float(a)
        except Exception:
            pass

    seg = ann.get("segmentation")
    if seg is None:
        return 0.0

    if isinstance(seg, list):  # polygons
        total = 0.0
        for poly in seg:
            coords = np.asarray(poly, dtype=float)
            if coords.size < 6:
                continue
            xs, ys = coords[0::2], coords[1::2]
            total += polygon_area(xs, ys)
        return total

    if isinstance(seg, dict) and PYCOCO_OK:  # RLE
        rle = seg
        if isinstance(rle.get("counts"), list):  # uncompressed
            rle = maskUtils.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]
        return float(maskUtils.area(rle))

    return 0.0


def union_mask_area(anns: List[Dict[str, Any]], img_h: int, img_w: int) -> float:
    """Union of masks (pixels). If pycocotools missing → fall back to SUM of areas."""
    if not PYCOCO_OK:
        return sum(seg_area_from_ann(a) for a in anns)

    R = None
    for ann in anns:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, list):  # polygons
            rles = maskUtils.frPyObjects(seg, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(seg, dict):
            rle = seg
            if isinstance(rle.get("counts"), list):
                rle = maskUtils.frPyObjects([rle], img_h, img_w)[0]
        else:
            continue
        R = rle if R is None else maskUtils.merge([R, rle])

    if R is None:
        return 0.0
    return float(maskUtils.area(R))


# ---------- core ----------

def collect_metrics(json_files: List[str], images_dir: str):
    per_ann = []    # rows for annotation_metrics.csv
    per_image = []  # rows for image_coverage_union.csv

    # Arrays for plots
    arr_bbox_area_over_image_area = []
    arr_bbox_w_over_image_w = []
    arr_bbox_h_over_image_h = []
    arr_bbox_aspect = []
    arr_image_coverage = []

    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco.get("images", [])}
        categories = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
        annotations = coco.get("annotations", [])

        # group anns by image
        by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            by_image.setdefault(ann["image_id"], []).append(ann)

        # --- per-annotation rows ---
        for ann in annotations:
            img = images.get(ann["image_id"], {})
            iw, ih, fn = img.get("width"), img.get("height"), img.get("file_name")
            if (not iw or not ih) and fn:
                try:
                    iw, ih = load_image_size(images_dir, fn)
                except Exception:
                    continue
            if not iw or not ih:
                continue

            iw, ih = float(iw), float(ih)
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            bx, by, bw, bh = bbox
            bw, bh = max(float(bw), 0.0), max(float(bh), 0.0)

            image_area = iw * ih
            row = {
                "source_json": os.path.basename(jp),
                "annotation_id": ann.get("id"),
                "image_id": ann.get("image_id"),
                "category_id": ann.get("category_id"),
                "category_name": categories.get(ann.get("category_id")),
                "image_file": fn,
                "image_width": int(iw),
                "image_height": int(ih),
                "bbox_x": float(bx),
                "bbox_y": float(by),
                "bbox_width": float(bw),
                "bbox_height": float(bh),
                "bbox_area": float(bw * bh),
                "bbox_area_over_image_area": safe_div(bw * bh, image_area),
                "bbox_w_over_image_w": safe_div(bw, iw),
                "bbox_h_over_image_h": safe_div(bh, ih),
                "bbox_aspect_ratio": safe_div(bw, bh),
                "iscrowd": ann.get("iscrowd"),
            }
            per_ann.append(row)

            # for plots
            arr_bbox_area_over_image_area.append(row["bbox_area_over_image_area"])
            arr_bbox_w_over_image_w.append(row["bbox_w_over_image_w"])
            arr_bbox_h_over_image_h.append(row["bbox_h_over_image_h"])
            arr_bbox_aspect.append(row["bbox_aspect_ratio"])

        # --- per-image (UNION) rows ---
        for img_id, img in images.items():
            anns = by_image.get(img_id, [])
            if not anns:
                continue

            iw, ih, fn = img.get("width"), img.get("height"), img.get("file_name")
            if (not iw or not ih) and fn:
                try:
                    iw, ih = load_image_size(images_dir, fn)
                except Exception:
                    continue
            if not iw or not ih:
                continue

            iw, ih = int(iw), int(ih)
            image_area = float(iw * ih)
            union_area = union_mask_area(anns, ih, iw)  # pixels
            coverage = safe_div(union_area, image_area)

            per_image.append({
                "source_json": os.path.basename(jp),
                "image_id": img_id,
                "image_file": fn,
                "image_width": iw,
                "image_height": ih,
                "num_annotations": len(anns),
                "union_mask_area_px": float(union_area),
                "image_area_px": float(image_area),
                "union_coverage_ratio": float(coverage),
                "coverage_mode": "union" if PYCOCO_OK else "sum_fallback"
            })
            arr_image_coverage.append(coverage)

    # numpy arrays for plotting
    arrays = {
        "per_ann_bbox_area_over_image_area": np.asarray(arr_bbox_area_over_image_area, dtype=float),
        "per_ann_bbox_w_over_image_w": np.asarray(arr_bbox_w_over_image_w, dtype=float),
        "per_ann_bbox_h_over_image_h": np.asarray(arr_bbox_h_over_image_h, dtype=float),
        "per_ann_bbox_aspect": np.asarray(arr_bbox_aspect, dtype=float),
        "per_image_coverage_ratio": np.asarray(arr_image_coverage, dtype=float),
    }
    return per_image, per_ann, arrays


def plot_hist(x: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: str,
              bins: int = 50, clip_hi_pct: float = None):
    if x.size == 0:
        print(f"[WARN] No data for {title}, skipping plot.")
        return
    data = x.copy()
    if clip_hi_pct is not None:
        hi = np.percentile(data, clip_hi_pct)
        data = np.clip(data, None, hi)
    counts, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    plt.figure(figsize=(8, 5))
    plt.bar(centers, counts, width=(edges[1] - edges[0]))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved plot: {out_path}")


def write_csv(rows: List[Dict[str, Any]], path: str, field_order: List[str] = None):
    if not rows:
        print(f"[WARN] No rows to write: {path}")
        return
    if field_order is None:
        # preserve a stable order
        field_order = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote CSV: {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Root folder for images.")
    ap.add_argument("--ann", required=True, help="COCO JSON file OR directory with JSONs.")
    ap.add_argument("--outdir", default="outputs", help="Folder for CSVs and plots.")
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    json_files = find_json_files(args.ann)

    per_image_rows, per_ann_rows, arrays = collect_metrics(json_files, args.images_dir)

    # ---- CSVs ----
    write_csv(
        per_image_rows,
        os.path.join(args.outdir, "image_coverage_union.csv"),
        field_order=[
            "source_json", "image_id", "image_file",
            "image_width", "image_height", "num_annotations",
            "union_mask_area_px", "image_area_px", "union_coverage_ratio",
            "coverage_mode",
        ],
    )

    write_csv(
        per_ann_rows,
        os.path.join(args.outdir, "annotation_metrics.csv"),
        field_order=[
            "source_json", "annotation_id", "image_id",
            "category_id", "category_name", "image_file",
            "image_width", "image_height",
            "bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_area",
            "bbox_area_over_image_area", "bbox_w_over_image_w",
            "bbox_h_over_image_h", "bbox_aspect_ratio", "iscrowd",
        ],
    )

    # ---- Plots (same as before) ----
    plot_hist(
        arrays["per_image_coverage_ratio"],
        title=f"Image coverage ratio ({'union' if PYCOCO_OK else 'sum_fallback'})",
        xlabel="union(ann areas)/image area" if PYCOCO_OK else "sum(ann areas)/image area",
        ylabel="image_count",
        out_path=os.path.join(args.outdir, "hist_image_coverage.png"),
        bins=args.bins,
    )
    plot_hist(
        arrays["per_ann_bbox_area_over_image_area"],
        title="bbox_area / image_area",
        xlabel="bbox_area / image_area",
        ylabel="bbox_count",
        out_path=os.path.join(args.outdir, "hist_bbox_area_over_image_area.png"),
        bins=args.bins,
        clip_hi_pct=99.0,
    )
    plot_hist(
        arrays["per_ann_bbox_w_over_image_w"],
        title="bbox_width / image_width",
        xlabel="bbox_width / image_width",
        ylabel="bbox_count",
        out_path=os.path.join(args.outdir, "hist_bbox_w_over_image_w.png"),
        bins=args.bins,
    )
    plot_hist(
        arrays["per_ann_bbox_h_over_image_h"],
        title="bbox_height / image_height",
        xlabel="bbox_height / image_height",
        ylabel="bbox_count",
        out_path=os.path.join(args.outdir, "hist_bbox_h_over_image_h.png"),
        bins=args.bins,
    )
    plot_hist(
        arrays["per_ann_bbox_aspect"],
        title="bbox_aspect_ratio (w/h)",
        xlabel="bbox_aspect_ratio (w/h)",
        ylabel="bbox_count",
        out_path=os.path.join(args.outdir, "hist_bbox_aspect_ratio.png"),
        bins=args.bins,
        clip_hi_pct=99.0,
    )


if __name__ == "__main__":
    main()
