"""
Demo 08: Dynamic reconstruction for a continuously rotating (helical) scan using MixedCubes.

This demo requires a custom dataset — set data_path to a directory containing:
  - geometry.yaml       scan geometry
  - projections.npy     projection stack [N, H, W]

The continuous scan mode reconstructs a scene that evolves during a single, uninterrupted
rotation rather than a sequence of full 360° sweeps.
"""

from pathlib import Path
import nect

data_path = Path("path/to/continious_scan_dataset")
geometry = nect.Geometry.from_yaml(data_path / "geometry.yaml")

reconstruction_path, _ = nect.reconstruct_continious_scan(
    geometry=geometry,
    projections=data_path / "projections.npy",
    quality="high",
    mode="dynamic",
    exp_name="dynamic_continious",
    config_override={
        "epochs": "5x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.001,
        "warmup": {
            "steps": 1400 * 10,
            "lr0": 0.001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 18,
            "n_features_per_level": 4,
            "log2_hashmap_size": 23,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "encoder_2d": {
            "n_levels": 12,
            "n_features_per_level": 4,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        },
        "net": {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 4,
            "include_identity": False,
        },
        "tv_spatial": 1e-4,
        "accumulation_steps": 4,
        "continous_scanning": True,
    },
    enc_arc="mixedcubes",
    memvstime="batch",
)
