## Hybrid Gaussian Splat/Mesh Renderer

Developed as part of the Advanced Computer Graphics course at FRI (UNI LJ).

A real-time renderer combining 3D Gaussian Splatting with mesh rasterisation. A configurable zone around the camera blends splat detail over a mesh base, with optional frequency-domain compositing and post-process depth of field.

### Report

[Seamless Transitions Between Gaussian Splatting and Mesh-Based Rendering](https://www.dropbox.com/scl/fi/5fw1rjunei65l3uf8w1on/Seamless_Transitions_Between_Gaussian_Splatting_and_Mesh_Based_Rendering.pdf?rlkey=7b9w7dckys3m5qy7kunlqk1ku&st=tnosoiz9&dl=0)

### Setup

Python 3.11 is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

### Models

Models are not included in the repository and must be downloaded separately ([Avaliable Here](https://www.dropbox.com/scl/fo/qanr7m3dllynoj1jyngtv/AKRkFwq79THC8cg0Dd3v9vE?rlkey=xqarcyy0lmbceupup9njvgxdv&st=qpw9gu36&dl=0)).

Place the models into the `models` folder.

### Running

```bash
cd hybrid_renderer
python main.py
```

Left-click drag to orbit, scroll to zoom.
