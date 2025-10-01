# Kintsugi-Dataset

This repository provides the **Kintsugi dataset**, a sustainable and aesthetic collection of ceramic repair data.  
Unlike conventional 3D fracture datasets that focus only on geometry, this dataset captures the **artistic and cultural aspects of kintsugi**, the traditional Japanese technique of repairing broken pottery with gold seams.  
It is designed to support research in **3D reconstruction, computer vision, cultural heritage preservation, and AI systems that go beyond recognition to embrace artistic meaning**.


---

## Dataset Contents

The dataset is organized into two main categories: **Artificial** and **Real**.  


```
Kintsugi_Data/
│
├── Artificial/                  # Artificially generated kintsugi-like plates
│   ├── circle_plate/            # Circle plate example
│   │   ├── circle_plate.ply
│   │   ├── piece_0.ply
│   │   └── piece_1.ply
│   ├── octangle_plate/
│   │   ├── octangle_plate.ply
│   │   └── piece_*.ply
│   └── square_plate/
│       ├── square_plate.ply
│       └── piece_*.ply
│
└── Real/                        
    ├── yobitsugi/               
    │   ├── yobitsugi_1/
    │   │   ├── yobitsugi_1.ply
    │   │   ├── yobitsugi_1.txt
    │   │   ├── piece_0.ply
    │   │   ├── piece_1.ply
    │   │   ├── piece_2.ply
    │   │   └── piece_3.ply    # 例として4ピース
    │   ├── yobitsugi_2/  ...
    │   └── yobitsugi_6/
    │
    └── kintsugi/
        ├── kintsugi_1/
        │   ├── kintsugi_1.ply
        │   ├── kintsugi_1.txt
        │   ├── piece_0.ply
        │   ├── piece_1.ply
        │   ├── piece_2.ply
        │   ├── piece_3.ply
        │   ├── piece_4.ply
        │   └── piece_5.ply    # 例として6ピース
        ├── kintsugi_2/ ...
        └── kintsugi_4/


```

### Notes
- Each folder contains:
  - A full repaired object point cloud (`*_n.ply`)  
  - A text file (`*_n.txt`) describing the repair process and artistic intent  
  - Fragment-level point clouds (`piece_*.ply`)  
- The number of fragments varies by case (e.g., *yobitsugi_1* has 4 pieces, *kintsugi_1* has 6 pieces).  
- Other cases follow the same structure.


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
  author       = {Nao Uematsu},
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

