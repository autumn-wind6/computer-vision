# Scripts layout

The command-line scripts are grouped by task so the project root is easier to browse:

```text
scripts/
  cloud_oneclick.sh     End-to-end server pipeline entry point.
  act/                  ACT training and offline evaluation.
  data/                 Dataset, capture, and COLMAP input preparation.
  render/               Fusion video rendering and frame extraction.
  setup/                Environment checks, workspace init, and smoke tests.
  utils/                Metrics collection, packaging, and final audits.
  vision/               2DGS, AIGC 3D generation, and PLY fusion.
```

Most scripts are meant to be run from the project root. For example:

```bash
python scripts/setup/check_environment.py --data-root /workspace/cv_final_data
python scripts/vision/merge_scene_assets.py --config configs/fusion_transforms.example.json --output /tmp/fusion.ply
python scripts/act/run_act_experiment.py train --config configs/act_env_b.json --output /tmp/act_env_b
```

Reusable Python functions live in `src/cvfinal/`; these scripts import that package when they need shared logic.
