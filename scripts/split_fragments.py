"""
split_fragments.py

Split a colored point cloud into individual fragments by:
1. Removing green kintsugi-line points
2. Estimating local thickness to determine eps
3. Running DBSCAN clustering
4. Refining oversized clusters with a second pass
5. Saving each fragment as piece_*.ply

---------------------------------------------
DBSCAN Parameter Selection (Design Rationale)
---------------------------------------------
We use a two-stage DBSCAN clustering strategy based on the estimated minimum
local thickness of the kintsugi repair lines.

Let `m` denote the minimal local diameter obtained from k-nearest neighbor
distance analysis. Because the thinnest region of the gold line corresponds to
the true physical separation between fragments, we define:

    First pass:  eps = m × 2
    Second pass: eps = m

The first pass (m × 2) provides a conservative clustering radius that ensures
large, coherent ceramic fragments are grouped together without unintended
over-segmentation. This produces stable high-level separation of fragments.

The second pass (m) is applied only to clusters that remain excessively large.
Using the smaller radius refines the boundaries within these regions and allows
DBSCAN to distinguish adjacent sub-fragments that were not separable at the
coarser scale.

This two-stage approach balances robustness and precision: the first pass
captures the global fragment structure, while the second pass resolves finer
local details in regions where fragments lie close to each other.

"""


import os
import sys
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------
# 1. Load point cloud
# ---------------------------------------------------------
def load_pointcloud(path: str):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load or empty point cloud: {path}")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


# ---------------------------------------------------------
# 2. Remove green points (kintsugi lines)
# ---------------------------------------------------------
def remove_green(points, colors):
    # HSV で緑色領域を抽出する
    rgb = (colors * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # 金継ぎ線の緑 (#0CEE45) 付近
    lower = np.array([35, 40, 40])     # H=35°, S=40, V=40
    upper = np.array([85, 255, 255])   # H=85°, S=255, V=255

    mask_green = cv2.inRange(hsv, lower, upper) > 0
    mask_nogreen = ~mask_green

    return points[mask_nogreen], colors[mask_nogreen]


# ---------------------------------------------------------
# 3. Estimate "thickness" to determine eps automatically
# ---------------------------------------------------------
def estimate_eps(points, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)

    # ignore distance to itself (0)
    mean_local = distances[:, 1:].mean(axis=1)
    local_d = mean_local * 2

    eps = float(local_d.min()) * 2
    return eps


# ---------------------------------------------------------
# 4. Run DBSCAN with fallback refinement
# ---------------------------------------------------------
def cluster_points(points, colors, eps, output_dir):
    labels = DBSCAN(eps=eps, min_samples=20).fit(points).labels_
    unique = set(labels)

    cluster_id = 0

    for lbl in unique:
        if lbl == -1:
            continue

        mask = (labels == lbl)
        pts = points[mask]
        cls = colors[mask]

        # If cluster is suspiciously big, refine with smaller eps
        if len(pts) > (len(points) / 2):
            sub_labels = DBSCAN(eps=eps / 2, min_samples=20).fit(pts).labels_
            for s_lbl in set(sub_labels):
                if s_lbl == -1:
                    continue
                sub_pts = pts[sub_labels == s_lbl]
                sub_cls = cls[sub_labels == s_lbl]

                save_piece(sub_pts, sub_cls, output_dir, cluster_id)
                cluster_id += 1
        else:
            save_piece(pts, cls, output_dir, cluster_id)
            cluster_id += 1

    print(f"Saved {cluster_id} fragments → {output_dir}")


# ---------------------------------------------------------
# 5. Save each fragment
# ---------------------------------------------------------
def save_piece(points, colors, output_dir, idx):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    filename = os.path.join(output_dir, f"piece_{idx}.ply")
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python split_fragments.py <input_color_ply> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)
    points, colors = load_pointcloud(input_path)
    points_nogreen, colors_nogreen = remove_green(points, colors)
    eps = estimate_eps(points_nogreen)
    print(f"[INFO] Estimated eps = {eps:.5f}")
    cluster_points(points_nogreen, colors_nogreen, eps, output_dir)


if __name__ == "__main__":
    import cv2
    main()
