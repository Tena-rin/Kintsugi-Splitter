"""
Minimal Working Pipeline for Kintsugi Splitter
Steps: Background removal → Kintsugi-line extraction → MoGe point cloud → DBSCAN fragments
"""
import os
import io
import cv2
import torch
import base64
import numpy as np
import open3d as o3d
from PIL import Image
from rembg import remove
from moge.model.v1 import MoGeModel
from google import genai
from google.genai import types
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from colorsys import rgb_to_hsv


# -----------------------
# Step 1: Background removal
# -----------------------
def step1_remove_background(inp, out):
    data = remove(open(inp, "rb").read())
    Image.open(io.BytesIO(data)).save(out)


# -----------------------
# Step 2: Kintsugi recoloring
# -----------------------
def step2_recolor_kintsugi(inp, out, api_key):
    genai.configure(api_key=api_key)
    client = genai.Client()

    prompt = (
        "Detect ONLY golden kintsugi repair lines in this image. "
        "Recolor them using a color NOT present in the plate. "
        "Use a uniform flat color (no shading). "
        "Do NOT modify the plate. "
        "Return PNG."
    )

    img = Image.open(inp).convert("RGB")

    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"]
        )
    )

    # Extract binary image output
    part = res.candidates[0].content.parts[0]
    binary = part.inline_data.data

    with open(out, "wb") as f:
        f.write(binary)


# -----------------------
# Helper: Detect recolor HSV range from edited image
# (Gemini の塗られた色を自動検出)
# -----------------------
def detect_recolor_hsv(original_img, recolored_img):
    ori = np.array(Image.open(original_img).convert("RGB")) / 255.0
    rec = np.array(Image.open(recolored_img).convert("RGB")) / 255.0

    diff = np.abs(ori - rec).mean(axis=2)
    mask = diff > 0.1   # 塗り替え部分推定

    recolor_pixels = rec[mask]
    if len(recolor_pixels) == 0:
        raise RuntimeError("Could not detect recolor area!")

    # RGB → HSV (0-255)
    hsv = cv2.cvtColor((recolor_pixels * 255).astype(np.uint8).reshape(-1, 1, 3),
                       cv2.COLOR_RGB2HSV).reshape(-1, 3)

    h_min, s_min, v_min = hsv.min(axis=0)
    h_max, s_max, v_max = hsv.max(axis=0)

    return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])


# -----------------------
# Step 3: MoGe depth → point cloud
# -----------------------
def step3_moge_to_pcd(inp, ply_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()

    img = cv2.cvtColor(cv2.imread(inp), cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(img / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    out = model.infer({"img": img_t})
    pts = out["points"].detach().cpu().numpy()[0]
    mask = out["mask"].detach().cpu().numpy()[0] > 0

    valid_pts = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_pts)
    o3d.io.write_point_cloud(ply_out, pcd)


# -----------------------
# Step 4: segment pieces
# -----------------------
def step4_segment_pieces(ply_in, hsv_lower, hsv_upper, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(ply_in)
    pts = np.asarray(pcd.points)

    # DBSCAN eps 自動推定
    nbrs = NearestNeighbors(n_neighbors=10).fit(pts)
    dist, _ = nbrs.kneighbors(pts)
    eps = np.median(dist[:, 1:]) * 2.0

    labels = DBSCAN(eps=eps, min_samples=20).fit_predict(pts)

    idx = 0
    for lbl in set(labels):
        if lbl == -1:
            continue
        frag = pts[labels == lbl]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(frag)
        o3d.io.write_point_cloud(f"{out_dir}/piece_{idx:03d}.ply", pc)
        idx += 1


# -----------------------
# Main entry
# -----------------------
if __name__ == "__main__":
    import sys

    inp = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY")

    bg = f"{outdir}/step1_removed.png"
    rec = f"{outdir}/step2_recolored.png"
    ply = f"{outdir}/step3_pointcloud.ply"

    print("Step 1: background remove")
    step1_remove_background(inp, bg)

    print("Step 2: recolor kintsugi")
    step2_recolor_kintsugi(bg, rec, api_key)

    print("Detecting recolor HSV range…")
    hsv_low, hsv_high = detect_recolor_hsv(bg, rec)
    print("Detected HSV range:", hsv_low, hsv_high)

    print("Step 3: MoGe → point cloud")
    step3_moge_to_pcd(bg, ply)

    print("Step 4: segmentation")
    step4_segment_pieces(ply, hsv_low, hsv_high, f"{outdir}/pieces")

    print("Pipeline complete!")
