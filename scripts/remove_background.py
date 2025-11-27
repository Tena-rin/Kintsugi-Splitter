"""
remove_background.py

Background removal utility using rembg (U²-Net).
Designed for use inside the Kintsugi-Dataset pipeline.

Usage:
    python scripts/remove_background.py input.png output.png
"""

import sys
import io
from rembg import remove
from PIL import Image


def remove_background(input_path: str, output_path: str) -> None:
    """Remove background from an image using rembg.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path where the processed image will be saved.
    """
    try:
        with open(input_path, "rb") as f:
            input_data = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_data = remove(input_data)
  
    output_image = Image.open(io.BytesIO(output_data))
    output_image.save(output_path)

    print(f"[OK] Background removed → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_background.py <input_image> <output_image>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    remove_background(in_path, out_path)
