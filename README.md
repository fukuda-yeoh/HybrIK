# HybrIK: Hybrid Analytical-Neural Inverse Kinematics for Body Mesh Recovery

<div align="center">
<img src="assets/hybrik_dance1.gif" height="160"> <img src="assets/hybrik_dance2.gif" height="160"> <img src="assets/hybrik_dance3.gif" height="160">
</br>
<img src="assets/hybrikx_dance1.gif" height="160"> <img src="assets/hybrikx_basketball.gif" height="160">
</div>

<div align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href='https://colab.research.google.com/drive/1n41l7I2NxWseuruVQEU8he2XqzSXhu2f?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>
<a href='https://paperswithcode.com/sota/3d-human-pose-estimation-on-agora?p=hybrik-x-hybrid-analytical-neural-inverse' style='padding-left: 0.5rem;'><img src='https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hybrik-x-hybrid-analytical-neural-inverse/3d-human-pose-estimation-on-agora'>
</a>
</br>
<a href="https://arxiv.org/abs/2011.14672">
    <img src='https://img.shields.io/badge/Paper-HybrIK-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
</a>
<a href="https://arxiv.org/abs/2304.05690">
    <img src='https://img.shields.io/badge/Paper-HybrIK--X-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=3CB371&color=40E0D0' alt='Paper PDF'>
</a>
<a href="https://youtu.be/xbdNFw2eZ4o">
    <img alt="youtube views" src="https://img.shields.io/badge/Demo-HybrIK-red?logo=youtube&logoColor=white&labelColor=D35400&style=for-the-badge"/>
</a>
<a href="https://youtu.be/VzCXKdQSZ7c">
    <img alt="youtube views" src="https://img.shields.io/badge/Demo-HybrIK--X-red?logo=youtube&logoColor=white&labelColor=D35400&style=for-the-badge"/>
</a>

</br>
</div>


This repo contains the code of our papers:

**HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation**, In CVPR 2021

**HybrIK-X: Hybrid Analytical-Neural Inverse Kinematics for Whole-body Mesh Recovery**, TPAMI 2025



<!-- ![hybrik](assets/hybrik.png) -->

<!-- 
<div align="center">
    <img src="assets/decompose.gif", width="600" alt><br>
    Twist-and-Swing Decomposition
</div> -->

## News :triangular_flag_on_post:
[2024/12/31] HybrIK-X is accepted by TPAMI!

[2023/06/02] Demo code for whole-body [HybrIK-X](https://arxiv.org/abs/2304.05690) is released.

[2022/12/03] HybrIK for Blender [add-on](https://github.com/Jeff-sjtu/HybrIK/releases/download/add-on/hybrik_blender_addon.zip) is now available for download. The output of HybrIK can be imported to Blender and saved as `fbx`.

[2022/08/16] [Pretrained model](https://drive.google.com/file/d/1C-jRnay38mJG-0O4_um82o1t7unC1zeT/view?usp=sharing) with HRNet-W48 backbone is available.

[2022/07/31] Training code with predicted camera is released.

[2022/07/25] [HybrIK](https://github.com/Jeff-sjtu/HybrIK) is now supported in [Alphapose](https://github.com/MVIG-SJTU/AlphaPose)! Multi-person demo with pose-tracking is available.

[2022/04/27] <a href='https://colab.research.google.com/drive/1n41l7I2NxWseuruVQEU8he2XqzSXhu2f?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a> is ready to use.

[2022/04/26] Achieve SOTA results by adding the 3DPW dataset for training.

[2022/04/25] The demo code is released!


## Key idea: Inverse Kinematics

HybrIK and HybrIK-X are based on a hybrid inverse kinematics (IK) to convert accurate 3D keypoints to parametric body meshes.

<div align="center">
    <img src="assets/ik.gif", width="600" alt><br>
    Twist-and-Swing Decomposition
</div>


## Installation instructions

HybrIK is a [uv](https://docs.astral.sh/uv/) project. PyTorch wheels for CPU,
NVIDIA CUDA, AMD ROCm and Intel XPU are provided as mutually exclusive extras
in `pyproject.toml`. `uv.lock` contains all of them; `uv sync` installs exactly
the backend you select. HybrIK supports Python 3.10-3.13. `.python-version`
pins 3.13 for standalone development because it is the newest release with
wheels for every backend, including the legacy `cu118` extra (CUDA 11.8 builds
stop at Python 3.13).

### Quick start

``` bash
# 1. Install uv (if you do not have it already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Pull our code
git clone https://github.com/Jeff-sjtu/HybrIK.git
cd HybrIK

# 3. Detect the PyTorch backend for this machine (read-only helper)
python3 scripts/detect_torch_backend.py

# 4. Install using the extra printed by the helper
uv sync --extra rocm7.2

# 5. Activate the environment
source .venv/bin/activate
```

`scripts/detect_torch_backend.py` only inspects the machine (`nvidia-smi`,
`rocm-smi`, `xpu-smi` and sysfs); it never creates environments or installs
packages. It prints the exact `uv sync` command to run. Use
`python3 scripts/detect_torch_backend.py --json` for machine-readable output,
or `--backend cpu` to force a backend. The same helper is available after
installation as `hybrik-detect-backend` and as `python -m hybrik.backend`.

| Hardware | Extra |
|----------|-------|
| NVIDIA GPU with a CUDA 11.8+ driver | `cu118`, `cu126`, `cu128`, `cu130` |
| AMD GPU (Linux) | `rocm6.4`, `rocm7.1`, `rocm7.2` |
| Intel GPU (Linux/Windows) | `xpu` |
| CPU, macOS, or no supported GPU | `cpu` |

Only one backend extra may be enabled at a time; `tool.uv.conflicts` enforces
this. If you prefer not to activate `.venv`, pass the extra to `uv run`:

``` bash
uv run --extra rocm7.2 python scripts/demo_image.py --img-dir examples --out-dir res
```

The `dev` dependency group is enabled by default, so `uv sync` also installs
pytest. ROCm wheels bundle their own runtime; use `rocm6.4` for older AMD
GPUs, or `rocm7.2` for current ones.

On integrated Radeon GPUs that report `gfx1152` (for example the Radeon 860M),
PyTorch may segfault on the first GPU operation because that ISA is not a
native ROCm target. Export a supported target before running:

``` bash
export HSA_OVERRIDE_GFX_VERSION=11.5.0
```

The detection helper prints a reminder when it sees `gfx1152`.

### PyTorch3D (optional: 3D mesh overlay only)

PyTorch3D is **optional**. It is used only to draw the 3D mesh overlay. Model
inference, the 2D skeleton video, `--save-pk` / `--save-pt` output, and SMPL-X
inference all run with plain PyTorch:

``` bash
uv run scripts/demo_video.py --video-name examples/dance.mp4 \
    --out-dir res_dance --save-pk
# -> res_dance/res_2d_dance.mp4 and res_dance/res.pk
```

When PyTorch3D is missing, the demo prints a notice, skips the overlay video
and `res_images/`, and still writes the files above. Installing PyTorch3D adds
the overlay video (`res_*.mp4`) and per-frame overlays.

PyTorch3D has no pre-built Linux wheel for current Python versions, so it is
compiled from source. Its build imports torch, so install it **after** the
PyTorch environment exists:

``` bash
uv sync --extra rocm7.2
uv pip install --no-build-isolation \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

This requires a C++ compiler and, for AMD builds, the ROCm development
packages (`hipcc` and the HIP headers).

PyTorch3D is intentionally not part of `uv.lock`, because its build imports
torch before the environment exists. A plain `uv sync` treats it as an
extraneous package and removes it, so use `--inexact` on later syncs:

``` bash
uv sync --extra rocm7.2 --inexact
```

SMPL-X rotation conversions now live in `hybrik.utils.rotation` and use only
PyTorch, so `SimpleTransform3DSMPLX`, the SMPL-X datasets, and
`scripts/demo_video_x.py` work without PyTorch3D. Only the overlay renderer
(`hybrik.utils.render_pytorch3d`) needs it; the SMPL-X demo falls back to the
2D video and `--save-pt` output when it is missing.

### Legacy OpenDR renderer

`hybrik/utils/render.py` is a legacy renderer that depends on OpenDR/Chumpy.
Those packages are unmaintained, require an older NumPy, and are not used by
the demos or training scripts, so they are not managed by this project. The
module is import-safe; calling it raises an `ImportError` explaining that
OpenDR is required. Use `hybrik/utils/render_pytorch3d.py` instead.

Chumpy is not needed to read the SMPL `.pkl` files: `hybrik.utils.smpl_pickle`
loads them with NumPy and SciPy only.

The video demos also require the `ffmpeg` executable to be available on
`PATH`.

Download the necessary model files from [[Google Drive](https://drive.google.com/file/d/1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1hVrUOt2QX_UTs4QuAgN2Lg?pwd=2u3c) (code: `2u3c`) ] and un-zip them in the `${ROOT}` directory. The licensed SMPL neutral model can also be fetched from the official SMPLify page:

``` bash
export SMPLIFY_USERNAME='you@example.com'
export SMPLIFY_PASSWORD='...'
export HYBRIK_MODEL_DIR=/path/to/model_files  # optional; defaults to ./model_files
./fetch_data.sh
```

The script is non-interactive and writes the model into `HYBRIK_MODEL_DIR`
(or `./model_files`).


## Using HybrIK as a library

HybrIK can be installed as a dependency of a larger project. The PyTorch
backend extras in this repository are for running HybrIK standalone; the
embedding project owns `torch`.

With uv, add it as a git dependency:

``` toml
[project]
dependencies = ["hybrik"]

[tool.uv.sources]
hybrik = { git = "https://github.com/Jeff-sjtu/HybrIK.git", rev = "<commit>" }
```

Pin a commit (or tag) for reproducible builds; `uv lock` in the embedding
project will then record the resolved revision. For local development, use a
path dependency instead:

``` bash
uv add --editable /path/to/HybrIK
# or, in pyproject.toml:
# [tool.uv.sources]
# hybrik = { path = "../HybrIK", editable = true }
```

The installed package exposes `hybrik`, `hybrik-detect-backend`, and the
`python -m hybrik.backend` module. A minimal smoke test is:

``` python
import hybrik
from hybrik import config_path

print(hybrik.__version__)
print(config_path("256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"))
```

Install the PyTorch build for your hardware in the embedding project as usual
(for example `uv pip install --torch-backend=auto torch torchvision`, or via
`tool.uv.sources` in your own `pyproject.toml`).

Licensed model files are not packaged. Point HybrIK at the directory that
contains them, either once:

``` bash
export HYBRIK_MODEL_DIR=/path/to/model_files
```

or at runtime:

``` python
import hybrik
hybrik.set_model_dir("/path/to/model_files")
```

The architecture configs are packaged, so a pretrained model can be built
without the HybrIK checkout:

``` python
from hybrik import config_path
from hybrik.models import builder
from hybrik.utils.config import update_config

cfg = update_config(
    config_path("256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml")
)
model = builder.build_sppe(cfg.MODEL)
```

`HYBRIK_CONFIG_DIR` (or `hybrik.set_config_dir(...)`) overrides the packaged
configs, and absolute paths are accepted by `update_config` as before.

Imports are lazy: `import hybrik`, `from hybrik.models import builder`,
`from hybrik.utils.presets import SimpleTransform3DSMPLCam`, and
`from hybrik.utils.presets import SimpleTransform3DSMPLX` do not import
PyTorch3D. PyTorch3D is only needed for the optional mesh overlay renderer
(`hybrik.utils.render_pytorch3d`) and demos that draw overlays.

To build a wheel instead of using a git dependency:

``` bash
uv build --out-dir dist
```

The wheel and sdist include `hybrik/configs/**/*.yaml`; model files are never
packaged. HybrIK supports Python 3.10-3.13; an embedding project can use any
interpreter in that range. The standalone `.python-version` pins 3.13 because
that is the newest interpreter supported by every PyTorch backend extra
(`cu118` is the limit).


## MODEL ZOO

### HybrIK (SMPL)

| Backbone | Training Data |     PA-MPJPE (3DPW)     | MPJPE (3DPW) | PA-MPJPE (Human3.6M) | MPJPE (Human3.6M) |  Download | Config |  
|----------|----------|------------|------------|-------|-----------|--------|--------------|
| ResNet-34          | w/ 3DPW | 44.5 | 72.4 | 33.8 | 55.5 | [model](https://drive.google.com/file/d/19ktHbERz0Un5EzJYZBdzdzTrFyd9gLCx/view?usp=share_link) | [cfg](./hybrik/configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml)    |
| HRNet-W48           | w/o 3DPW | 48.6 | 88.0 | 29.5 | 50.4 | [model](https://drive.google.com/file/d/1o3z99bebm2XImElc3XEUzTNVhQboGJE9/view?usp=share_link) | [cfg](./hybrik/configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml)    |
| HRNet-W48          | w/ 3DPW | 41.8 | 71.3 | 29.8 | 47.1 | [model](https://drive.google.com/file/d/1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW/view?usp=share_link) | [cfg](hybrik/configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml)    |

### HybrIK-X (SMPL-X)
| Backbone |     MVE (AGORA Test)     | MPJPE (AGORA Test) |  Download | Config |  
|----------|------------|------------|-------|--------------|
| HRNet-W48           | 134.1 | 127.5 | [model](https://drive.google.com/file/d/1bKIPD60z_Im4S3W2-rew6YtOtUGff6-v/view?usp=sharing) | [cfg](hybrik/configs/smplx/256x192_hrnet_smplx_kid.yaml)    |
| HRNet-W48 + [RLE](https://github.com/Jeff-sjtu/res-loglikelihood-regression/tree/203dc3195ee5a11ed6f47c066ffdb83247511359)          | 112.1 | 107.6 | [model](https://drive.google.com/file/d/1R0WbySXs_vceygKg_oWeLMNAZCEoCadG/view?usp=sharing) | [cfg](hybrik/configs/smplx/256x192_hrnet_rle_smplx_kid.yaml)    |


## Demo
Download the pretrained model (with predicted camera) and place it in
`./pretrained_models/`, i.e. `./pretrained_models/hybrik_hrnet.pth` and
`./pretrained_models/hybrikx_rle_hrnet.pth`. Pass `--checkpoint` to use a
different location.

The SMPL model files must also be available. Point HybrIK at them with
`HYBRIK_MODEL_DIR` (recommended, and required when installed as a dependency):

``` bash
export HYBRIK_MODEL_DIR=/path/to/model_files
```

For a source checkout you can also symlink them into the repository root;
`model_files/` is gitignored:

``` bash
ln -s /path/to/model_files ./model_files
```

Every demo accepts `--config` (a packaged config name or path) and
`--checkpoint` (a checkpoint path).

### SMPL

* Visualize HybrIK on **videos** (run in single frame) and save results:

``` bash
uv run python scripts/demo_video.py --video-name examples/dance.mp4 --out-dir res_dance --save-pk --save-img
```
The saved results in `./res_dance/res.pk` can be imported to Blender with our [add-on](https://github.com/Jeff-sjtu/HybrIK/releases/download/add-on/hybrik_blender_addon.zip).

* Visualize HybrIK on **images** (requires PyTorch3D for the mesh overlay):

``` bash
uv run python scripts/demo_image.py --img-dir examples --out-dir res
```

### SMPL-X

``` bash
uv run python scripts/demo_video_x.py --video-name examples/dance.mp4 --out-dir res_dance --save-pt --save-img
```
This writes `./res_dance/res.pt` and the 2D video. The 3D overlay is added
when PyTorch3D is installed.

## Fetch data
Download *Human3.6M*, *MPI-INF-3DHP*, *3DPW* and *MSCOCO* datasets. You need to follow directory structure of the `data` as below. Thanks to the great job done by Moon *et al.*, we use the Human3.6M images provided in [PoseNet](https://github.com/mks0601/3DMPPE_POSENET_RELEASE).
```
|-- data
`-- |-- h36m
    `-- |-- annotations
        `-- images
`-- |-- pw3d
    `-- |-- json
        `-- imageFiles
`-- |-- 3dhp
    `-- |-- annotation_mpi_inf_3dhp_train.json
        |-- annotation_mpi_inf_3dhp_test.json
        |-- mpi_inf_3dhp_train_set
        `-- mpi_inf_3dhp_test_set
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- train2017
        `-- val2017
```
* Download Human3.6M parsed annotations. [ [Google](https://drive.google.com/drive/folders/1tLA_XeZ_32Qk86lR06WJhJJXDYrlBJ9r?usp=sharing) | [Baidu](https://pan.baidu.com/s/1bqfVOlQWX0Rfc0Yl1a5VRA) ]
* Download 3DPW parsed annotations. [ [Google](https://drive.google.com/drive/folders/1f7DyxyvlC9z6SFT37eS6TTQiUOXVR9rK?usp=sharing) | [Baidu](https://pan.baidu.com/s/1d42QyQmMONJgCJvHIU2nsA) ]
* Download MPI-INF-3DHP parsed annotations. [ [Google](https://drive.google.com/drive/folders/1Ms3s7nZ5Nrux3spLxmMMAQWc5aAIecmv?usp=sharing) | [Baidu](https://pan.baidu.com/s/1aVBDudbDRT1w_ZxQc9zicA) ]


## Train from scratch

``` bash
./scripts/train_smpl_cam.sh test_3dpw hybrik/configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml
```

## Evaluation
Download the pretrained model ([ResNet-34](https://drive.google.com/file/d/16Y_MGUynFeEzV8GVtKTE5AtkHSi3xsF9/view?usp=sharing) or [HRNet-W48](https://drive.google.com/file/d/1C-jRnay38mJG-0O4_um82o1t7unC1zeT/view?usp=sharing)).
``` bash
./scripts/validate_smpl_cam.sh ./hybrik/configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml ./pretrained_hrnet.pth
```


## Citing
If our code helps your research, please consider citing the following paper:

    @inproceedings{li2021hybrik,
        title={Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation},
        author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={3383--3393},
        year={2021}
    }

    @article{li2023hybrik,
        title={HybrIK-X: Hybrid Analytical-Neural Inverse Kinematics for Whole-body Mesh Recovery},
        author={Li, Jiefeng and Bian, Siyuan and Xu, Chao and Chen, Zhicun and Yang, Lixin and Lu, Cewu},
        journal={arXiv preprint arXiv:2304.05690},
        year={2023}
    }
