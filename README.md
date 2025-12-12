# Kintsugi Splitter
Digitally preserving the cultural and artistic value of kintsugi  
**by Tena-rin**

Kintsugi Splitter is a project dedicated to **preserving the cultural, artistic, and philosophical value of kintsugi**,  
the Japanese technique of repairing broken ceramics with gold.

Rather than treating breakage as damage, kintsugi embraces it as a part of the object’s history.  
Kintsugi Splitter continues this philosophy in the digital era by using **AI-based reconstruction, segmentation,  
and multimodal analysis** to archive repaired objects and make their stories accessible worldwide.

![Pipeline Overview](IMG_5200.webp)

---

## Purpose

Modern digitization often captures only geometry or appearance.  
However, **kintsugi is not merely a physical repair technique—it is a philosophy** rooted in:

- *sustainability*
- *beauty in imperfection* 
- *human intention and emotion* 

**Kintsugi Splitter aims to preserve not only the shape,  
but also the cultural essence of kintsugi and share it globally.**

By combining:

- background removal  
- AI-based kintsugi-line extraction  
- 3D reconstruction using MoGe  
- fragment segmentation via DBSCAN  
- and structured narrative metadata generation  

the project creates a new form of digital cultural archive.


---
## Why Open Source?

We release Kintsugi Splitter openly so that people around the world  
can learn through kintsugi the value of repairing, cherishing,  
and sustaining a single object rather than replacing it.

This project aims to preserve not only techniques but also the  
cultural philosophy of kintsugi for future generations.

---
## Example Output
<div style="width:100%; display:flex; flex-direction:row; justify-content:space-between; align-items:flex-start; gap:20px;">

  <!-- 左：元画像 -->
  <div style="flex:1; text-align:center;">
    <h3>Input Image</h3>
    <img src="example_dataset/images/kintsugi.png" style="width:25%; max-width:300px;">
  </div>

  <!-- 中央：GIF（3D動画） -->
  <div style="flex:1; text-align:center;">
    <h3> Output: 3D point cloud </h3>
    <img src="example_dataset/videos/kintsugi_demo.gif" style="width:100%; max-width:300px;">
  </div>

  <!-- 右：プロンプト -->
  <div style="flex:1;">
    <h3> Output: Prompt </h3>
    <pre style="white-space:pre-wrap; font-size:14px;">

A. Metadata (Objective Information)
├── 1. Material / Texture
│     └── Porcelain (ceramics); smooth and elegant surface
├── 2. Kintsugi Line Color
│     └── Gold
├── 3. Brand / Origin
│     ├── Imari-yaki (Hizen region, Japan)
│     ├── Made in Japan
│     └── Item No.: 9884-500-0027-9100
├── 4. Production Period
│     └── Estimated Edo period
├── 5. Dimensions / Weight
│     ├── Size: Φ15 cm × H 5 cm
│     └── Weight: 305 g
└── 6. Artist
      └── Repaired by Iyo Kimura, Atelier fourteen


B. Story (Subjective Information)
├── 1. How it Was Broken
│     └── Damaged during the Noto earthquake; received from an Imari collector
├── 2. Artistic Considerations
│     └── Kintsugi lines were made subtle and gentle to respect original patterns
└── 3. Feelings Behind the Repair
      └── “A fresh dressing” was given with hope that the piece gains a renewed future
  </div>

</div>

---

## Pipeline Overview

The Kintsugi Splitter pipeline consists of five stages:

1. **Background removal**  
   Removes photographic context to isolate the ceramic object.

2. **Kintsugi-line extraction (AI-assisted)**  
   Detects gold repair lines and generates precise masks for analysis.

3. **3D point cloud generation (MoGe)**  
   Reconstructs the object into a 3D point cloud with color.

4. **Fragment segmentation**  
   Removes gold lines and separates fragments using a two-stage DBSCAN method  
   based on local thickness estimation.

5. **Metadata & story creation**  
   Captures *material*, *origin*, *crafting environment*, and *artistic intent*  
   to preserve the cultural meaning behind the repair.

This multi-step pipeline allows Kintsugi Splitter to preserve both the **form** and the **spirit**  
of repaired objects.

![Pipeline Overview](pipeline.png)

---

## Usage (Underconstruction)

We opensourced each steps python code and are working on providing minimum pipline code.
This pipline will be provided soon!!!

---

### 1. Install dependencies

```bash
pip install open3d
pip install opencv-python
pip install pillow
pip install rembg
pip install onnxruntime
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/microsoft/MoGe.git
pip install google-generativeai
pip install scikit-learn
```

---

### 2. Set your Gemini API key

Windows (PowerShell):

```powershell
setx GEMINI_API_KEY "YOUR_API_KEY"
```

---

### 3. Prepare an input image

Place your kintsugi photo in the working directory:

```
input.png
```

---


## Dataset Types

Kintsugi Splitter works with two kinds of input images:

### Artificial Data (人工データ)
These images are **not actually broken nor repaired**.  
They are intact plates where **kintsugi-like lines were manually drawn using digital tools**  
(e.g., Photoshop) to simulate fracture patterns and gold repairs.  
This allows testing under controlled conditions.

### Real Kintsugi Data (実データ)
Images of **actual repaired ceramic works**, including yobitsugi and traditional kintsugi plates.

![Real Example](Artificial&Real.png)



---

## Vision

Kintsugi Splitter is part of a larger initiative to create a **sustainable digital archive**  
that honors Japanese craftsmanship and spreads its values—including resilience, repair,  
and imperfect beauty—to a global audience.

By transforming physical objects into structured digital artifacts,  
we aim to build a foundation for future research in:

- cultural computing  
- AI-driven heritage preservation  
- multimodal understanding  
- sustainable design  

Kintsugi teaches that breakage is not an end, but a beginning.  
Kintsugi Splitter brings that message to the world.

---

## License

### Dataset (Images, Masks, Point Clouds, Metadata)
This dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

When using the dataset, please include the following attribution:

> “Kintsugi 3D Dataset. Object courtesy of Atelier fourteen (木村いよ).”

Full license: https://creativecommons.org/licenses/by/4.0/

---

### Code (Python scripts / tools / notebooks)
All source code in this repository is released under the **MIT License**.  
See `LICENSE_CODE` for details.


---

## Dissemination

Parts of this project were presented at:

- Information Processing Society of Japan (IPSJ), 2025  
- Science Conference 2025  
- JSEC 2025 (Japan Science & Engineering Challenge)

We release this project to help share the philosophy of kintsugi with the world  
and encourage a culture of repairing, valuing, and sustaining objects.

---

## Acknowledgements

I would like to express my deepest gratitude to Ms. Iyo Kimura, director of Atelier fourteen, for her generous guidance on kintsugi techniques and aesthetics, as well as for kindly granting permission to use her kintsugi works in this study.

---

##  References

### Background Removal
- Qin, X., Zhang, Z., Huang, C., Dehghani, M., Zaidi, S., Qin, Z., & Hou, Q.  
  **U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection**.  
  *Pattern Recognition*, 2020.  
  https://github.com/NathanUA/U-2-Net

### 3D Reconstruction (MoGe)
- Ruicheng, W., et al.  
  **MoGe: Accurate Monocular Geometry Estimation for Open-Domain Images**.  
  *CVPR*, 2025.  
  https://github.com/microsoft/MoGe

### Clustering (Fragment Segmentation)
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X.  
  **A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN)**.  
  *KDD*, 1996.

### Multimodal Modeling (Google Gemini)
- Google DeepMind  
  **Gemini 2.5 Flash Model Card**, 2025.  
  https://ai.google.dev/gemini/

### Background Removal Tool
- danielgatis  
  **rembg: Image Background Removal**  
  GitHub Repository: https://github.com/danielgatis/rembg

### Kintsugi Artist / Studio  
- Atelier fourteen  
  **Atelier fourteen (Official Website)**  
  Website: https://www.atelierfourteen.com/  
  Accessed: 2025-08-31  
  Object courtesy
  
---

## Citations

If you find **Kintsugi Splitter** useful in your research or projects, please cite:

```bibtex
@misc{Kintsugi-Splitter2025,
    title        = {Kintsugi Splitter},
    author       = {Nao Uematsu},
    year         = {2025},
    note         = {GitHub Repository},
    url          = {https://github.com/Tena-rin/Kintsugi-Splitter}
}
```




