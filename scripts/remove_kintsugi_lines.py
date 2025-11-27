"""
remove_kintsugi_lines.py

Use Google Gemini to detect and paint kintsugi lines in a uniform green color.
The result is saved as an output image (PNG).

Usage:
    python scripts/remove_kintsugi_lines.py input.png output.png
"""

import sys
import base64
from google import generativeai as genai


def load_image_as_bytes(path: str) -> bytes:
    """Read image as raw bytes."""
    with open(path, "rb") as f:
        return f.read()


def save_base64_image(base64_data: str, output_path: str):
    """Save a base64-returned image to a file."""
    img_bytes = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(img_bytes)


def extract_kintsugi_lines(input_path: str, output_path: str, api_key: str):
    """Send the input image to Gemini and retrieve the painted-line output."""
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")  # or newer model

    prompt = (
        "Please detect the golden repair lines (kintsugi lines) in this ceramic plate "
        "and paint them in a flat solid green (#0CEE45), with no shading or texture. "
        "Do NOT paint the plate itself — only the repair lines."
    )

    img_bytes = load_image_as_bytes(input_path)

    response = model.generate_content(
        [
            prompt,
            {"mime_type": "image/png", "data": img_bytes}
        ],
        generation_config={"response_mime_type": "image/png"}
    )

    # Gemini returns image in base64
    base64_img = response.candidates[0].content.parts[0].text

    save_base64_image(base64_img, output_path)

    print(f"[OK] Kintsugi lines extracted → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_kintsugi_lines.py <input_image> <output_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Get API key (環境変数に入れておくべき!!)
    import os
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Please set environment variable: GEMINI_API_KEY")

    extract_kintsugi_lines(input_path, output_path, api_key)
