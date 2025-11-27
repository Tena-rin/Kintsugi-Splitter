# Kintsugi-Dataset Pipeline

This repository provides the **processing pipeline** used to generate the Kintsugi 3D point cloud dataset  
introduced in the associated research report.

> ⚠️ **Note (Important):**  
> The original dataset (images, artworks, and point clouds) is **not publicly available**  
> due to copyright and artistic ownership restrictions.  
>  
> This repository releases **only the code, workflow, and documentation** required  
> to reproduce the pipeline using your own images.

Our goal is to support research in  
**3D reconstruction, computer vision, cultural heritage preservation,  
and AI systems that can understand artistic and cultural meaning**.

---

## Pipeline Overview

The pipeline automatically converts a single photo of a kintsugi or yobitsugi plate  
into fragment-level 3D point clouds.

It consists of the following steps:

1. **Background removal** (U²-Net / rembg)  
2. **Kintsugi-line extraction** using Google Gemini  
3. **3D point cloud generation** (MoGe model)  
4. **Gold-line removal + DBSCAN-based fragment segmentation**


### Notes
- Each case contains:
  - A full repaired object point cloud (`*_n.ply`)
  - A repair description text (`*_n.txt`)
  - Fragment-level point clouds (`piece_*.ply`)
- The number of fragments varies by case.
- This structure is **automatically generated** when running the pipeline.

---

## Usage

### Load a Point Cloud and Description File

```python
import open3d as o3d

# Load sample point cloud
pcd = o3d.io.read_point_cloud("examples/sample_output.ply")
o3d.visualization.draw_geometries([pcd])

# Load sample description
with open("examples/sample_description.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text)

