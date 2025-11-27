"""
generate_pointcloud.py

Generate a colored 3D point cloud from a single image using MoGe.
MoGe (Microsoft Research) is licensed under MIT License.

Usage:
    python scripts/generate_pointcloud.py input.png output.ply
"""

import sys
import cv2
import torch
import numpy as np
import open3d as o3d
from moge.model.v1 import MoGeModel


def load_image(path: str):
    """Load image with OpenCV and convert to normalized RGB FloatTensor."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
    return tensor, img


def to_device():
    """Choose CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_pointcloud(input_path: str, output_path: str):
    """Run MoGe and save a colored point cloud."""
    device = to_device()

    # Load image
    img_tensor, img_rgb = load_image(input_path)
    img_tensor = img_tensor.to(device)

    # Load MoGe model
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        output = model.infer(img_tensor)

    # Extract data
    points = output["points"].detach().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()

    # Remove batch dimension if present
    if points.ndim == 4:
        points = points.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    mask = mask > 0  # Boolean mask

    # Resize original image to match model output
    H, W = mask.shape
    img_resized = cv2.resize(img_rgb, (W, H))

    # Extract valid points and colors
    points_valid = points[mask]
    colors_valid = img_resized[mask]

    # Create colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_valid)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)

    # Save
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_pointcloud.py <input_image> <output_ply>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_pointcloud(input_path, output_path)
