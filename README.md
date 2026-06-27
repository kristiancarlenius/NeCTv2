# NeCTv2: Neural Computed Tomography v2

NeCTv2 is an extended and improved implementation of NeCT, developed as a master's thesis at the Norwegian University of Science and Technology (NTNU) in collaboration with the CT lab at Equinor. It uses implicit neural representations (INR) — powered by [`tiny-cuda-nn`](https://github.com/NVlabs/tiny-cuda-nn/) — to reconstruct CT volumes from raw projection data, supporting both static 3D CT and dynamic 4D CT.

<table>
  <tr>
    <td>
      <img src="docs/images/showcase1.gif" width="480">
      <p>
        Rendering of spontaneous imbibition in a Bentheimer sandstone reconstructed using NeCT. The brine flowing into the sample is shown in light blue, while the salt grains dissolving are presented in red.
      </p>
    </td>
    <td>
      <img src="docs/images/showcase2.gif" width="480">
      <p>
        Rendering of the dissolution of a salt grain. Three orthogonal slices visualize its temporal evolution. In the xz slice, it is possible to observe the brine coming into contact with the salt before it starts to dissolve.
      </p>
    </td>
  </tr>
</table>

<p align="center">
    <a href="https://github.com/kristiancarlenius/NeCTv2" target="_blank">
        <img src="https://img.shields.io/badge/NeCTv2%20Repository-blueviolet?style=for-the-badge&logo=github" alt="NeCTv2 Repository"/>
    </a>
</p>

- [What's new in v2](#whats-new-in-v2)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Models](#models)
- [Demo](#demo)
- [Data](#data)
- [GUI](#gui)
- [Licensing and Citation](#licensing-and-citation)

![NeCT Reconstruction Pipeline](docs/images/pipeline.png)

---

## What's new in v2

NeCTv2 builds on the original NeCT with a set of architectural and workflow improvements developed during a master's thesis:

- **Extended model zoo** — New dynamic architectures: `quadcubes`, `sexcubes`, `singlecube`, `combinedcubes`, `mixedcubes`, and transformer/U-Net hybrid variants (`quadcubes_transformer`, `sexcubes_unet`, etc.)
- **Static-to-dynamic initialization** (`IniTrainer`) — Pre-train a fast static `hash_grid` model and transfer its weights into a `quadcubes` dynamic model, giving the dynamic reconstruction a warm start and faster convergence
- **Continuous scanning support** (`reconstruct_continious_scan`) — Dedicated trainer for helical / continuously rotating scan geometries
- **Zarr export** (`export_volume_zarr`) — Compressed chunked volume export in addition to TIFF
- **Richer configuration** — Fine-grained control over `w0` warm-up, gradient accumulation (`accumulation_steps`), dampening factors (`damp_multi`), and more
- **Improved sampling** — Adaptive detector downsampling schedule and flexible points-per-ray curriculum during training

---

## Installation

NeCTv2 has been tested on **Windows** and **Linux** with the following dependencies:

| Package         | Version           | Notes              |
|-----------------|-------------------|--------------------|
| python          | 3.11 \| 3.12      |                    |
| pytorch         | 2.4 – 2.7         |                    |
| CUDA            | 12.X              |                    |
| CMake (Linux)   | 3.24              | For tiny-cuda-nn   |
| C++17 (Windows) |                   | For tiny-cuda-nn   |

> **Recommended:** Use [conda](https://docs.anaconda.com/free/anaconda/install/) or [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage your Python environment.
>
> - For video export with the `avc1` codec, use conda. With uv, video export falls back to `mp4v`.
> - Tested with `python=3.11, 3.12` and `pytorch>=2.4,<2.8`.

**Note:** Ensure `PATH` and `LD_LIBRARY_PATH` include the CUDA binaries as described in [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/). Building binaries for both `tiny-cuda-nn` and NeCTv2 may take several minutes.

### uv

```bash
uv venv --python=3.12
source venv/bin/activate          # Windows: venv\Scripts\activate
uv pip install -e .[torch]
uv pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
```

#### Custom PyTorch version

Visit the [PyTorch Installation Page](https://pytorch.org/get-started/locally/) to install a specific version, then:

```bash
uv pip install -e . --no-build-isolation-package torch
uv pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
```

### conda

```bash
conda create -n nectv2 python=3.12 -y
conda activate nectv2
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 lightning==2.1 conda-forge::opencv -c pytorch -c nvidia -c conda-forge -y
pip install -e .
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Multiple CUDA compute capabilities

Set these environment variables **before** installing to build for multiple GPU generations (60=P100, 70=V100, 80=A100, 90=H100):

```bash
export CUDA_ARCHITECTURES="60;70;80;90"
export CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
export TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
export TORCH_CUDA_ARCH_LIST="6.0 7.0 8.0 9.0"
export FORCE_CUDA="1"
```

---

## Quick start

### Static CT

```python
import numpy as np
import nect

geometry = nect.Geometry(
    DSD=1500.0,
    DSO=1000.0,
    nDetector=[256, 512],
    dDetector=[1.75, 1.75],
    nVoxel=[256, 512, 256],
    dVoxel=[1.0, 1.0, 1.0],
    angles=np.linspace(0, 360, 49, endpoint=False),
    mode="cone",
    radians=False,
)

volume = nect.reconstruct(
    geometry=geometry,
    projections="path/to/projections.npy",  # [nProjections, height, width] or a directory of images
    quality="medium",
)
np.save("volume.npy", volume)
```

### Dynamic (4D) CT

```python
import nect

geometry = nect.Geometry.from_yaml("path/to/geometry.yaml")

reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections="path/to/projections.npy",
    quality="high",
    mode="dynamic",
    exp_name="my_experiment",
)
nect.export_video(reconstruction_path, add_scale_bar=True, acquisition_time_minutes=60)
```

### Static-to-dynamic initialization (NeCTv2)

Pre-train a static `hash_grid` model, then warm-start a dynamic `quadcubes` model from it:

```python
import nect

geometry = nect.Geometry.from_yaml("path/to/geometry.yaml")

# Step 1 – static reconstruction
static_path, _ = nect.reconstruct(
    geometry=geometry,
    projections="path/to/projections.npy",
    quality="high",
    mode="static",
    exp_name="static_init",
)

# Step 2 – dynamic reconstruction initialized from the static model
nect.reconstruct(
    geometry=geometry,
    projections="path/to/projections.npy",
    quality="high",
    mode="dynamic",
    exp_name="dynamic_from_static",
    static_init=static_path,
)
```

---

## Models

### Static

| Model       | Description                                    |
|-------------|------------------------------------------------|
| `hash_grid` | Multi-resolution hash grid (default, fast)     |
| `kplanes`   | K-Planes decomposition                         |
| `tricubes`  | Triplane hash grid variant                     |

### Dynamic (4D CT)

| Model                     | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `quadcubes`               | Four-cube multi-resolution hash grid for spacetime (primary dynamic model)  |
| `sexcubes`                | Six-cube variant for higher-capacity spacetime encoding                     |
| `singlecube`              | Lightweight single-cube dynamic model                                       |
| `combinedcubes`           | Combined static + dynamic hash grid                                         |
| `mixedcubes`              | Mixed resolution cubes                                                      |
| `double_hash_grid`        | Dual hash grid for dynamic scenes                                           |
| `kplanes_dynamic`         | K-Planes with temporal planes                                               |
| `hypercubes`              | Hypercube grid for high-dimensional encoding                                |
| `quadcubes_transformer`   | QuadCubes with transformer decoder                                          |
| `quadcubes_unet`          | QuadCubes with U-Net decoder                                                |
| `sexcubes_transformer`    | SexCubes with transformer decoder                                           |
| `sexcubes_unet`           | SexCubes with U-Net decoder                                                 |

---

## Demo

Demo scripts are in the [`demo/`](./demo/) folder. Projection data is downloaded automatically on first run.

| Script | Description |
|--------|-------------|
| `00_static_reconstruct_from_file.py` | Static reconstruction from a `.npy` file |
| `01_static_reconstruct_from_array.py` | Static reconstruction from a NumPy array |
| `02_geometry_load.py` | Load geometry from YAML |
| `03_dynamic_reconstruction_video.py` | Dynamic reconstruction with video export |
| `04_dynamic_reconstruction_export_volume.py` | Dynamic reconstruction with volume export |
| `05_parallel_beam.py` | Parallel beam geometry |
| `06_export_video_projections.py` | Export video from projections |
| `07_reconstruct_from_cfg_file.py` | Reconstruction via config file |
| `08_static_trainer.py` | Advanced static trainer configuration |
| `09_continious_scan.py` | Continuous scanning reconstruction |

---

## Data

All projection data from the dynamic experiments are available at [Zenodo](https://zenodo.org/records/16448474).

---

## GUI

The NeCT GUI uses PyQt5 (GPL licensed) and lives in a separate repository.

---

## Licensing and Citation

This project is licensed under the **MIT License**.

A master's thesis project at NTNU in collaboration with the CT lab at Equinor.

If you use NeCTv2 in your research, please cite: **(Will be added)**
