from PIL import Image, ImageOps
import os
import argparse
import csv
from pathlib import Path

def flip_image_variants(image: Image.Image):
    """
    Given a PIL Image, return a dictionary with:
    - 'original': the original image
    - 'flip_x': horizontally flipped (left↔right)
    - 'flip_y': vertically flipped (top↔bottom)
    - 'flip_xy': both flipped
    """
    flip_x = ImageOps.mirror(image)
    flip_y = ImageOps.flip(image)
    flip_xy = ImageOps.flip(flip_x)

    return {
        'original': image,
        'flip_x': flip_x,
        'flip_y': flip_y,
        'flip_xy': flip_xy
    }

def flip_label(label, flip_x=False, flip_y=False):
    """
    Flip a (rel_x, rel_y, rel_z) label according to image flip.
    """
    x, y, z = label
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return (x, y, z)

def augment_folder_with_flips(dataset_dir: str):
    dataset_dir = Path(dataset_dir)
    labels_csv = dataset_dir / "labels.csv"

    # New output directory
    out_dir = dataset_dir.parent / f"{dataset_dir.name}_augmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "labels.csv"

    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "rel_x", "rel_y", "rel_z"])

        for row in rows:
            img_path = dataset_dir / row["filename"]
            image = Image.open(img_path).convert("RGB")
            base_name = Path(row["filename"]).stem
            label = (float(row["rel_x"]), float(row["rel_y"]), float(row["rel_z"]))

            variants = flip_image_variants(image)
            for key, img in variants.items():
                suffix = "" if key == "original" else f"_{key}"
                filename = f"{base_name}{suffix}.png"
                img.save(out_dir / filename)

                flipx = "x" in key
                flipy = "y" in key
                flipped_label = flip_label(label, flip_x=flipx, flip_y=flipy)
                writer.writerow([filename, *flipped_label])

    print(f"✅ Augmented dataset saved to: {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flip and augment dataset with label updates")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder with labels.csv and images")
    args = parser.parse_args()

    augment_folder_with_flips(args.dataset)
