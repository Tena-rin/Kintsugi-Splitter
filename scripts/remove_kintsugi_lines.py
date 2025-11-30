"""
remove_kintsugi_lines.py

Use Google Gemini 2.5 Flash to detect and paint kintsugi lines in a uniform green color.
The result is saved as an output image (PNG).

Usage:
    python scripts/remove_kintsugi_lines.py input.png output.png
"""

import sys
import base64
from google import generativeai as genai


def load_image_as_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def save_base64_image(base64_data: str, output_path: str):
    img_bytes = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(img_bytes)


def extract_kintsugi_lines(input_path: str, output_path: str, api_key: str):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = (
        "Detect the golden kintsugi repair lines in the plate image and recolor them as a flat solid green (#0CEE45). "
        "The plate itself must not be recolored—only the repair lines. "
        "Return the result as a PNG image."
    )

    img_bytes = load_image_as_bytes(input_path)

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/png",
                "data": img_bytes
            }
        ],
        generation_config={
            "response_mime_type": "image/png"
        }
    )

    base64_img = response.text  # .text contains the base64 data for images

    save_base64_image(base64_img, output_path)

    print(f"[OK] Kintsugi lines extracted → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_kintsugi_lines.py <input_image> <output_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    import os
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY in your environment variables.")

    extract_kintsugi_lines(input_path, output_path, api_key)
