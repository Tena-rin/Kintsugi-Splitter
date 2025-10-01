# Kintsugi-Dataset
This repository contains the **Kintsugi 3D dataset**, a collection of broken ceramic objects represented as point clouds, meshes, and masks.  
The dataset is designed for research in **3D reconstruction, computer vision, and cultural heritage restoration**.

---

## Dataset Contents

The dataset is organized as follows:

```
kintsugi-3d-dataset/
│
├── images/           # Input RGB images
├── masks/            # Binary masks highlighting fracture or repair regions
├── pointclouds/      # 3D point clouds in PLY format
├── meshes/           # Mesh reconstructions (OBJ/STL)
├── metadata.json     # Metadata and annotations
└── README.md         # This file
```

- **Images**: High-resolution photographs of broken and repaired objects  
- **Masks**: Pixel-level annotations of cracks or gold repair lines  
- **Pointclouds**: Point sets extracted from multi-view or depth data  
- **Meshes**: 3D reconstructed surfaces for downstream tasks  
- **Metadata**: Object ID, capture conditions, and annotation details  

---

## Usage

Example: Loading a point cloud with Python (Open3D):

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("pointclouds/bowl_01.ply")
o3d.visualization.draw_geometries([pcd])
```

Example: Using a mask with OpenCV:

```python
import cv2

mask = cv2.imread("masks/bowl_01_mask.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("mask", mask)
cv2.waitKey(0)
```

---

## Citation

If you use this dataset in your research, please cite:

```
@dataset{kintsugi_3d_2025,
  author       = {Your Name and Collaborators},
  title        = {Kintsugi 3D Dataset},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/yourname/kintsugi-3d-dataset}
}
```

---

## License

This dataset is released under the [CC BY 4.0 License](LICENSE).  
You are free to share and adapt the material, but must give appropriate credit.

---

## Acknowledgements

This dataset was created as part of ongoing research on  
**digital kintsugi, 3D reconstruction, and cultural heritage preservation**.  
We thank the contributors and supporting institutions for their help.

