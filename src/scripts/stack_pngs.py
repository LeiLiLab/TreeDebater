"""Stack PNG images vertically (top-to-bottom) into a single image."""
import argparse
from pathlib import Path
from PIL import Image


def stack_vertical(image_paths, out_path):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    width = max(im.width for im in images)
    height = sum(im.height for im in images)
    combined = Image.new("RGB", (width, height), "white")
    y = 0
    for im in images:
        x = (width - im.width) // 2
        combined.paste(im, (x, y))
        y += im.height
    combined.save(out_path)
    print(f"Saved → {out_path}  ({len(images)} images, {width}x{height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="PNG files to stack top-to-bottom")
    parser.add_argument("-o", "--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    stack_vertical([Path(p) for p in args.inputs], Path(args.output))
